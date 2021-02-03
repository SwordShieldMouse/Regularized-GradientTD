import numpy as np
from numpy.linalg import norm
import sys

from src.utils import Averages

from src.agents.BaseAgent import BaseAgent
from src.agents.OLO import *
from src.agents import OLO

class ParameterFree(BaseAgent):
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

        self.z = np.zeros(features)
        self.lmda = params.get("lambda", 0.0)
        self.avg_t = params.get("averaging", "Uniform")

    def update(self, x, a, xp, r, gamma, rho):

        g_theta, g_y  = self.grads(x, a, xp, r, gamma, rho)
        self._apply(g_theta, g_y)

    def _apply(self, g_theta, g_y):
        self.theta.update(g_theta); self.y.update(g_y)
        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.av_theta.update(self.theta_t); self.av_y.update(self.y_t)

    def grads(self, x, a, xp, r, gamma, rho):
        if self.lmda > 0:
            self.z = gamma * self.lmda * rho * self.z + x

            v = self.theta_t.dot(x)
            vp=self.theta_t.dot(xp)
            g = r + gamma * vp
            delta = r + gamma*vp - v

            gtheta = - rho * self.z.dot(self.y_t)*(x-gamma*xp)
            gy = - rho*delta*self.z + x.dot(self.y_t)*x

            return gtheta, gy

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
        self.theta.initWeights(u)
        self.theta_t = self.theta.bet()
        self.av_theta.reset(self.theta_t)

    def _initBets(self):
        avg_t = getattr(Averages, self.avg_t)
        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.av_theta, self.av_y = avg_t(self.theta_t), avg_t(self.y_t)

class PFGTD(ParameterFree):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

        # opt params
        W0 = params['wealth'] / 2
        self.theta = Param(features, W0, params["hint"], params["beta"])
        self.y = Param(features, W0, params["hint"], params["beta"])

        self._initBets()

class PFGTDVectorHints(ParameterFree):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

        # opt params
        W0 = params['wealth'] / 2
        self.theta = VectorHintsParam(features, W0, params["hint"], params["beta"])
        self.y = VectorHintsParam(features, W0, params["hint"], params["beta"])

        self._initBets()

class CWPFGTD(ParameterFree):
    """
    Coordinate-wise Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)

        W0 = params['wealth'] / 2

        # opt params
        self.theta = CWParam(features, W0, params["hint"], params["beta"])
        self.y = CWParam(features, W0, params["hint"], params["beta"])

        self._initBets()

class PFCombined(ParameterFree):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        p = self._params(params)

        algA = getattr(sys.modules[__name__], params['algA'])
        self.A = algA(features, actions, p)

        algB = getattr(sys.modules[__name__], params['algB'])
        self.B = algB(features, actions, p)

    def bet(self):
        theta_t = self.A.theta.bet()+self.B.theta.bet()
        y_t = self.A.y.bet() + self.B.y.bet()
        return theta_t, y_t

    def update(self, x, a, xp, r, gamma, rho):
        self.theta_t, self.y_t = self.bet()

        gtheta, gy = self.grads(x, a, xp, r, gamma, rho)
        self.A._apply(gtheta, gy)
        self.B._apply(gtheta, gy)

    def getWeights(self):
        return self.A.getWeights() + self.B.getWeights()

    def initWeights(self, u):
        self.A.initWeights(u/2)
        self.B.initWeights(u/2)

    def _params(self, params):
        # For now let's just assume that both players get the same
        # wealth
        p = params.copy()
        p['wealth'] = params['wealth'] / 2
        return p

class PFGTDPlus(PFCombined):
    def __init__(self, features, actions, params):
        params["algA"] = "PFGTDVectorHints"
        params["algB"] = "CWPFGTD"
        super().__init__(features, actions, params)
