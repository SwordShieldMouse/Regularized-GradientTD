import numpy as np
from numpy.linalg import norm

from agents.OLO import Param, SCParam, CWParam

class ParameterFree:
    def __init__(self, features: int, params: dict):
        self.features = features
        self.params = params
        self.gamma = params['gamma']

        # Average decisions
        self.av_theta = np.zeros(features)
        self.av_y = np.zeros(features)

        self.t = 0.0

    def update(self, x, a, r, xp, rho):
        self.t += 1

        # get bets
        theta_t = self.theta.bet()
        y_t = self.y.bet()

        self.av_theta += 1.0 / self.t * (theta_t - self.av_theta)
        self.av_y += 1.0 / self.t * (y_t - self.av_y)

        # construct gradients
        #
        # ================================
        # --- EFFICIENT IMPLEMENTATION ---
        # ================================
        #  implicitly compute A to avoid
        #  the outerproduct op
        # --------------------------------
        d = x - self.gamma * xp
        g_theta = - rho * d * np.dot(x, y_t)
        g_y = rho * x * np.dot(d, theta_t) - rho * r * x + x * np.dot(x, y_t)

        # ===========================
        # --- Slow implementation ---
        # ===========================
        # useful for debugging
        # ---------------------------
        # d = x - self.gamma * xp
        # At = rho * np.outer(x, d)
        # bt = rho* r * x
        # Mt = np.outer(x, x)
        #
        # g_theta = np.matmul(- At.transpose(), y_t)
        # g_y = np.matmul(At, theta_t) - bt + np.matmul(Mt, y_t)

        self.theta.update(g_theta)
        self.y.update(g_y)

    def getWeights(self):
        return self.av_theta

    def setInitialBet(self, u):
        raise(NotImplementedError('setInitialBet not implemented'))

class PFGTD(ParameterFree):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, params: dict):
        super().__init__(features, params)
       
        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features, params["wealth"], params["hint"], params["beta"])

    def setInitialBet(self, u):
        u = np.array(u, dtype='float64')
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm
        self.theta.beta = 1.0
        self.av_theta = self.theta.bet()

class SCPFGTD(PFGTD):
    """
    Parameter-free GTD with hints and curvature adaptation in
    the dual variables y
    """
    def __init__(self, features: int, params: dict):
        super().__init__(features,params)

        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = SCParam(features, params["wealth"], params["hint"], params["beta"])


class CWPFGTD(ParameterFree):
    """
    Coordinate-wise Parameter-free GTD with hints
    """
    def __init__(self, features: int, params: dict):
        super().__init__(features, params)

        # opt params
        self.theta = CWParam(features, params["wealth"], params["hint"], params["beta"])
        self.y = CWParam(features, params["wealth"], params["hint"], params["beta"])

    def setInitialBet(self, u):
        u = np.array(u, dtype='float64')
        self.theta.W = u
        self.theta.beta = 1.0
        self.av_theta = self.theta.bet()

