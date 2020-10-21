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
        self.z = np.zeros(features)

    def applyUpdate(self, s, a, sp, r, gamma):
        g_theta, g_y  = self.grads(s,a,sp,r,gamma,self._rho(a,s))
        self._apply(g_theta, g_y, a)

        return None, None

    def _rho(self, a, s):
        mu = self.policy(s, self.getWeights())
        pi = self.policy(s, self.theta_t)
        return pi[a] / mu[a]

    def grads(self, s, a, sp, r, gamma, rho):
        # Default grads = GTD2

        x = self.rep.encode(s,a)
        xp = self.rep.encode(sp, argmax(self.Qs(sp, self.theta_t)))

        q_a = self.theta_t.dot(x)
        qp_m = self.theta_t.dot(xp)

        g = r + gamma * qp_m
        delta = g - q_a

        self.z = gamma * self.lmda * rho * self.z + x

        gy = - rho*delta*self.z + x.dot(self.y_t)*x
        gtheta = - rho * self.z.dot(self.y_t)*(x-gamma*xp)

        return gtheta, gy

    def _apply(self, g_theta, g_y, a):
        self.theta.update(g_theta)
        self.y.update(g_y)

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
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.bet()

        avg_t = getattr(Averages, params.get('averaging', "Uniform"))
        self.av_theta, self.av_y = avg_t(self.theta_t), avg_t(self.y_t)

    def bet(self):
        return self.theta.bet(), self.y.bet()

class PFGQ2(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def _rho(self, a, s):
        mu = self.policy(s, self.getWeights())
        return 1.0/mu[a] if np.argmax(self.Q(s,a,self.theta_t)) == a else 0.0

class EPFGQ(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def grads(self, x, a, xp, r, gamma, rho):
        # Default grads = GTD2

        pi_t = self.policy(x)
        eqp = self.theta_t.dot(xp).dot(pi_t)
        q_a = self.theta_t[a].dot(x)

        g = r + gamma * eqp
        delta = g - q_a

        #TODO: Unsure about this
        self.z *= gamma*self.lmda*pi_t[a]
        self.z[a] += x

        gy_a = - delta*self.z[a] + x.dot(self.y_t[a])*x
        gtheta_a = - self.z[a].dot(self.y_t[a])*(x-gamma*xp)

        gtheta= np.zeros((self.actions, self.features))
        gy = np.zeros((self.actions, self.features))
        gtheta[a,:]= gtheta_a
        gy[a, :] = gy_a

        return gtheta.flatten(), gy.flatten()

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
