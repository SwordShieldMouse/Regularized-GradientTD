import numpy as np
from numpy.linalg import norm

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

        gy_a = - rho*delta*self.z[a] + x.dot(self.y_t[a])*x
        gtheta_a = - rho * self.z[a].dot(self.y_t[a])*(x-gamma*xp)

        gtheta= np.zeros((self.actions, self.features))
        gy = np.zeros((self.actions, self.features))
        gtheta[a,:]= gtheta_a
        gy[a, :] = gy_a

        return gtheta.flatten(), gy.flatten()

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
        self.theta = Param(features * actions, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features * actions, params["wealth"], params["hint"], params["beta"])

        self.theta_t, self.y_t = self.bet()

        avg_t = getattr(Averages, params.get('averaging', "Uniform"))
        self.av_theta, self.av_y = avg_t(self.theta_t), avg_t(self.y_t)

    def bet(self):
        theta_t = self.theta.bet().reshape(self.actions, self.features)
        y_t = self.y.bet().reshape(self.actions,self.features)
        return theta_t, y_t

    def initWeights(self, theta, y):
        theta = np.array(theta, dtype='float64').flatten()
        y = np.array(y, dtype='float64').flatten()

        self.theta.initWeights(theta)
        self.y.initWeights(y)

        self.theta_t, self.y_t = self.bet()
        self.av_theta.reset(self.theta_t)
        self.av_y.reset(self.y_t)

class BatchPFGQ(PFGQ):
    """
    Parameter-free GTD with hints; single learner for all actions
    """
    def __init__(self, features: int, actions: int, params: dict):
        super().__init__(features, actions, params)
        self.buff = []
        self.buffsz = params.get('buffer_size', 0)
        self.sequential = params.get('sequential', False)
        self.endOfEpisode = params.get('endOfEpisode', False)
        assert not (self.endOfEpisode and self.buffsz > 0)

    def applyUpdate(self, x, a, xp, r, gamma):
        self.buff.append((x,a, xp, r, gamma))
        if (self.endOfEpisode and gamma==0) or len(self.buff) == self.buffsz :
            indices = range(len(self.buff)) if self.sequential else np.random.permutation(len(self.buff))
            for idx in indices:
                xi, ai, xpi, ri, gammai = self.buff[idx]
                g_theta, g_y  = self.grads(xi, ai, xpi, ri, gammai,self._rho(ai,xi))
                self._apply(g_theta, g_y, ai)
            self.buff = []

        return None, None

class BatchPFGQReset(BaseAgent):
    """
    Parameter-free GTD with hints; single learner for all actions
    """
    def __init__(self, features: int, actions: int, params: dict):
        params['alpha'] = None
        super().__init__(features, actions, params)

        self.buff = []
        self.buffsz = params.get('buffer_size', 1)

        self.agent = PFGQ(features, actions, params)
        self.theta = self.agent.getWeights()
        self.av_theta = Averages.LastIterate(self.theta)

    def getWeights(self):
        return self.av_theta.get()

    def applyUpdate(self, x, a, xp, r, gamma):
        self.buff.append((x,a, xp, r, gamma))
        if len(self.buff) == self.buffsz:
            indices = np.random.permutation(self.buffsz)
            for idx in indices:
                self.agent.applyUpdate(*self.buff[idx])

            # Maybe want to average?
            self.theta = self.agent.getWeights()
            self.av_theta.update(self.theta)
            theta_t, y_t = self.agent.av_theta.get(), self.agent.av_y.get()
            self.agent = PFGQ(self.features, self.actions, self.params)
            self.agent.initWeights(theta_t, y_t)
            self.buff = []

        return None, None

class PFGQ2(PFGQ):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def _rho(self, a, x):
        mu = self.policy(x, self.getWeights())
        return 1.0/mu[a] if np.argmax(self.theta_t.dot(x)) == a else 0.0

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
