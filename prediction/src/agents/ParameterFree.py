import numpy as np
from numpy.linalg import norm

from src.utils import Averages

from src.agents.BaseAgent import BaseAgent
from src.agents.OLO import Param, SCParam, CWParam, ParamUntrunc

class ParameterFree(BaseAgent):
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

    def update(self, x, a, xp, r, gamma, rho):

        g_theta, g_y  = self.grads(x, a, xp, r, gamma, rho)
        self._apply(g_theta, g_y)

    def _apply(self, g_theta, g_y):
        self.theta.update(g_theta); self.y.update(g_y)
        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.av_theta.update(self.theta_t); self.av_y.update(self.y_t)

    def grads(self, x, a, xp, r, gamma, rho):
        # ================================
        # --- EFFICIENT IMPLEMENTATION ---
        # ================================
        #  implicitly compute A to avoid
        #  the outerproduct op
        # --------------------------------
        d = x - gamma * xp
        g_theta = - rho * d * np.dot(x, self.y_t)
        g_y = rho * x * np.dot(d, self.theta_t) - rho * r * x + x * np.dot(x, self.y_t)

        # ===========================
        # --- Slow implementation ---
        # ===========================
        # useful for debugging
        # ---------------------------
        # d = x - gamma * xp
        # At = rho * np.outer(x, d)
        # bt = rho* r * x
        # Mt = np.outer(x, x)
        #
        # g_theta = np.matmul(- At.transpose(), y_t)
        # g_y = np.matmul(At, theta_t) - bt + np.matmul(Mt, y_t)
        return g_theta, g_y

    def getWeights(self):
        return self.av_theta.get()

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm
        self.theta.beta = 1.0
        self.av_theta.reset(self.theta.bet())

class PFGTD(ParameterFree):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)
       
        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()

        avg_t = getattr(Averages, params.get('averaging','Uniform'))
        self.av_theta, self.av_y = avg_t(self.theta_t), avg_t(self.y_t)

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm
        self.theta.beta = 1.0
        self.av_theta.reset(self.theta.bet())

class PFGTDUntrunc(PFGTD):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)
        self.theta = ParamUntrunc(features, params["wealth"], params["hint"], params["beta"])
        self.y = ParamUntrunc(features, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.av_theta.reset(self.theta_t); self.av_y.reset(self.y_t)

class PFTDC(PFGTD):
    """
    Parameter-free TDC with hints
    """
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

    def grads(self, x, a, xp, r, gamma, rho):
        # Default grads = GTD2
        delta = r + gamma * np.dot(theta_t,xp) - np.dot(theta_t,x)
        g_theta = - rho * delta * x + rho*gamma*np.dot(y_t,theta_t)*xp

        d = x - gamma * xp
        g_y = -rho * delta + np.dot(y_t,x) * x #rho * x * np.dot(d, theta_t) - rho * r * x + x * np.dot(x, y_t)
        return g_theta, g_y

class SCPFGTD(PFGTD):
    """
    Parameter-free GTD with hints and curvature adaptation in
    the dual variables y
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features,actions,params)

        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = SCParam(features, params["wealth"], params["hint"], params["beta"])


class CWPFGTD(ParameterFree):
    """
    Coordinate-wise Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)

        # opt params
        self.theta = CWParam(features, params["wealth"], params["hint"], params["beta"])
        self.y = CWParam(features, params["wealth"], params["hint"], params["beta"])

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        self.theta.W = u
        self.theta.beta = 1.0
        self.av_theta.reset(self.theta.bet())
