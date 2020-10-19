import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent
from src.agents.OLO import Param, ParamUntrunc
from src.utils import Averages

class ParameterFree(BaseAgent):
    def __init__(self, features, actions, params):
        params['alpha'] = None
        super().__init__(features, actions, params)

        self.lmda = params.get('lambda', 0.0)
        self.z = np.zeros((actions, features))

    def applyUpdate(self, x, a, xp, r, gamma):
        g_theta, g_y  = self.grads(x,a,xp,r,gamma,self._rho(a,x))
        self._apply(g_theta, g_y, a)

        return None, None

    def _rho(self, a, x):
        mu = self.policy(x, self.getWeights())
        pi = self.policy(x, self.theta_t)
        return pi[a] / mu[a]

    def grads(self, x, a, xp, r, gamma, rho):
        # Default grads = GTD2

        q_a = self.theta_t[a].dot(x)
        qp_m = self.theta_t.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a

        self.z *= gamma * self.lmda * rho
        self.z[a] += x

        dy_a = - rho*delta*self.z[a] + x.dot(self.y_t[a])*x
        dtheta_a = - rho * self.z[a].dot(self.y_t[a])*(x-gamma*xp)

        gtheta= np.zeros((self.actions, self.features))
        gy = np.zeros((self.actions, self.features))
        gtheta[a,:]= gtheta_a
        gy[a, :] = gy_a

        return gtheta.flatten(), dh.flatten()

    def _apply(self, g_theta, g_y, a):
        self.theta.update(gtheta)
        self.y.update(gy)

        self.theta_t, self.y_t = self.bet()
        self.av_theta.update(self.theta_t); self.av_y.update(self.y_t)

    def getWeights(self):
        return self.av_theta.get()

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

        avg_t = getattr(Averages, params.get('averaging', "Uniform"))
        self.av_theta, self.av_y = avg_t(self.theta_t), avg_t(self.y_t)

    def bet(self):
        theta_t = self.theta.bet().reshape(self.actions, self.features)
        y_t = self.y.bet().reshape(self.actions,self.features)
        return theta_t, y_t

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm
        self.theta.beta = 1.0
        self.av_theta.reset(self.theta.bet())

class PFGQ2(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def _rho(self, a, x):
        mu = self.policy(x, self.getWeights())
        return 1.0/mu[a] if np.argmax(self.theta_t.dot(x)) == a else 0.0

class UncorrectedPFGQ(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def grads(self, x, a, xp, r, gamma, rho):
        # Default grads = GTD2

        pi = self.policy(x)
        eqp = self.theta_t.dot(xp).dot(pi)
        q_a = self.theta_t[a].dot(x)

        g = r + gamma * eqp
        delta = g - q_a

        self.z *= gamma*self.lmda*pi
        self.z[a] += x

        dy_a = - delta*self.z[a] + x.dot(self.y_t[a])*x
        dtheta_a = - self.z[a].dot(self.y_t[a])*(x-gamma*xp)

        gtheta= np.zeros((self.actions, self.features))
        gy = np.zeros((self.actions, self.features))
        gtheta[a,:]= gtheta_a
        gy[a, :] = gy_a

        return gtheta.flatten(), dh.flatten()

class PFGQUntrunc(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.theta = ParamUntrunc(features * actions, params["wealth"], params["hint"], params["beta"])
        self.y = ParamUntrunc(features * actions, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.bet()
        self.av_theta.reset(self.theta_t); self.av_y.reset(self.y_t)

class PFGQScaledGrad(PFGQ):
    """
    Parameter-free GTD with hints; single learner for all actions
    """
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def applyUpdate(self, x, a, xp, r, gamma):
        g_theta, g_y  = self.grads(x,a,xp,r,gamma, self._rho(a,x))
        g_theta /= np.linalg.norm(x)
        g_y /= np.linalg.norm(x)
        self._apply(g_theta, g_y, a)
