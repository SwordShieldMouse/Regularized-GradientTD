import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent
from src.agents.OLO import Param, ParamUntrunc
from src.utils import Averagers

class ParameterFree(BaseAgent):
    def __init__(self, features, actions, params):
        params['alpha'] = None
        super().__init__(features, actions, params)

    def policy(self, x, w):
        max_acts = argmax(w.dot(x))
        pi = np.zeros(self.actions)
        uniform = self.epsilon / self.actions
        pi += uniform

        # if there are no max acts, then we've diverged and everything is NaNs
        if len(max_acts) > 0:
            pi[max_acts] += (1.0 / len(max_acts)) - self.epsilon

        return pi

    def selectAction(self, x):
        return sample(self.policy(x, self.getWeights()))

    def applyUpdate(self, x, a, xp, r, gamma):
        g_theta, g_y  = self.grads(x,a,xp,r,gamma)
        self._update(g_theta, g_y, a)

        self.theta_t, self.y_t = self.bet()

        self.av_theta.update(self.theta_t)
        self.av_y.update(self.y_t)

    def grads(self, x, a, xp, r, gamma):
        # Default grads = GTD2

        mu = self.policy(x, self.getWeights())
        pi = self.policy(x, self.theta_t)
        rho = pi[a] / mu[a]

        # mu = self.policy(x, self.av_theta)
        # rho = 1.0/mu[a] if np.argmax(self.theta_t.dot(x)) == a else 0.0

        q_a = self.theta_t[a].dot(x)

        qp_m = self.theta_t.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a

        delta_hat = self.y_t[a].dot(x)

        dw = - rho * x.dot(self.y_t[a])*(x-gamma*xp)
        dh = - rho*delta*x + x.dot(self.y_t[a])*x

        return dw, dh

    def getWeights(self):
        return self.av_theta.get()

    def _update(self, gtheta, gy):
        raise(NotImplementedError("_update not implemented"))

    def bet(self):
        raise(NotImplementedError('bet not implemented'))

    def initWeights(self, u):
        raise(NotImplementedError('setInitialBet not implemented'))


class PFGQ(ParameterFree):
    """
    Parameter-free GTD with hints; single learner for all actions
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)

        # opt params
        self.theta = Param(features * actions, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features * actions, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.bet()

        averager = getattr(Averagers, params['averager'])
        self.av_theta, self.av_y = averager(self.theta_t), averager(self.y_t)

    def bet(self):
        theta_t = self.theta.bet().reshape(self.actions, self.features)
        y_t = self.y.bet().reshape(self.actions,self.features)
        return theta_t, y_t

    def _update(self, g_theta, g_y, a):
        gtheta = np.zeros((self.actions, self.features))
        gy = np.zeros((self.actions, self.features))

        gtheta[a,:]=g_theta
        gy[a, :] = g_y
        self.theta.update(gtheta.flatten())
        self.y.update(gy.flatten())

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm
        self.theta.beta = 1.0
        self.av_theta.reset(self.theta.bet())

class PFGQUntrunc(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.theta = ParamUntrunc(features * actions, params["wealth"], params["hint"], params["beta"])
        self.y = ParamUntrunc(features * actions, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.bet()

        averager = getattr(Averagers, params['averager'])
        self.av_theta, self.av_y = averager(self.theta_t), averager(self.y_t)

class PFGQScaledGrad(PFGQ):
    """
    Parameter-free GTD with hints; single learner for all actions
    """
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def applyUpdate(self, x, a, xp, r, gamma):
        g_theta, g_y  = self.grads(x,a,xp,r,gamma)
        g_theta /= np.linalg.norm(x)
        g_y /= np.linalg.norm(x)
        self._update(g_theta, g_y, a)

        self.theta_t, self.y_t = self.bet()

        self.av_theta.update(self.theta_t)
        self.av_y.update(self.y_t)
