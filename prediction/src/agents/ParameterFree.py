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

        g_theta, self.h_theta = self.constrain_and_clip(g_theta, self.h_theta, self.theta_t, self.theta_t_hat)
        g_y, self.h_y = self.constrain_and_clip(g_y, self.h_y, self.y_t, self.y_t_hat)

        self._apply(g_theta, g_y)

    def constrain_and_clip(self, g, h, w, hatw):
        gtrunc, h = self.clip(g, h)

        vt = (w-hatw)/norm(w-hatw) if norm(w)>self.D else np.zeros_like(w)
        gtilde = gtrunc if np.dot(gtrunc, w) >= np.dot(gtrunc,hatw) else gtrunc-np.dot(gtrunc,vt)*vt

        return gtilde, h


    def clip(self, g, h):
        raise NotImplementedError("ParameterFree.clip not implemented")

    def _apply(self, g_theta, g_y):

        self.theta.update(g_theta, self.h_theta)
        self.y.update(g_y, self.h_y)

        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.theta_t_hat, self.y_t_hat = self.proj(self.theta_t), self.proj(self.y_t)
        self.av_theta.update(self.theta_t_hat)
        self.av_y.update(self.y_t_hat)

    def grads(self, x, a, xp, r, gamma, rho):
        # ================================
        # --- EFFICIENT IMPLEMENTATION ---
        # ================================
        #  implicitly compute A to avoid
        #  the outerproduct op
        # --------------------------------
        d = x - gamma * xp
        g_theta = - rho * d * np.dot(x, self.y_t_hat)
        g_y = rho * x * np.dot(d, self.theta_t_hat) - rho * r * x + x * np.dot(x, self.y_t_hat)

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
        self.theta_t_hat = self.proj(self.theta_t)
        self.av_theta.reset(self.theta_t_hat)

    def _initBets(self):
        avg_t = getattr(Averages, self.avg_t)
        self.theta_t, self.y_t = self.theta.bet(), self.y.bet()
        self.theta_t_hat, self.y_t_hat = self.proj(self.theta_t), self.proj(self.y_t)
        self.av_theta, self.av_y = avg_t(self.theta_t_hat), avg_t(self.y_t_hat)

class PFGTD(ParameterFree):
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions, params: dict):
        super().__init__(features, actions, params)

        # opt params
        W0 = params['wealth'] / 2
        self.theta = Param(features, W0, params["beta"])
        self.y = Param(features, W0, params["beta"])

        self.h_theta = params['hint']
        self.h_y = params['hint']

        self._initBets()

    def clip(self, g, h):

        # Incorporate grad bound
        gradnorm = norm(g)
        gtrunc = g if gradnorm < h else h*g / gradnorm
        h = max(h, gradnorm)
        return gtrunc, h

class CWPFGTD(ParameterFree):
    """
    Coordinate-wise Parameter-free GTD with hints
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)

        W0 = params['wealth'] / 2

        # opt params
        self.theta = CWParam(features, W0, params["beta"])
        self.y = CWParam(features, W0, params["beta"])

        self.h_theta = params['hint'] * np.ones(features)
        self.h_y = params['hint'] * np.ones(features)

        self._initBets()

    def clip(self, g, vec_h):

        # Incorporate grad bound
        gradnorm = np.abs(g)

        gtrunc = g.copy()
        truncIdx = np.argwhere(gradnorm > vec_h)
        gtrunc[truncIdx] = np.multiply(vec_h[truncIdx], g[truncIdx]) / gradnorm[truncIdx]
        vec_h = np.maximum(vec_h, gradnorm)
        return gtrunc, vec_h


class PFGTDPlus(ParameterFree):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        W0 = params['wealth']/2
        self.theta = PFPlus(features, W0, params['beta'])
        self.y = PFPlus(features, W0, params['beta'])

        self.h_theta = np.ones(features)*params['hint']
        self.h_y = np.ones(features)*params['hint']

        self._initBets()

    def clip(self, g, vec_h):

        # Incorporate grad bound
        gradnorm = np.abs(g)

        gtrunc = g.copy()
        truncIdx = np.argwhere(gradnorm > vec_h)
        gtrunc[truncIdx] = np.multiply(vec_h[truncIdx], g[truncIdx]) / gradnorm[truncIdx]
        vec_h = np.maximum(vec_h, gradnorm)
        return gtrunc, vec_h
