import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent
from src.utils import Averages

class GQ(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.lmda = params.get('lambda', 0.0)
        self.z = np.zeros((actions,features))

        self.theta = np.zeros((actions, features))
        self.y = np.zeros((actions, features))

        avg_t = getattr(Averages, params.get('averaging', "Uniform"))
        self.av_theta, self.av_y = avg_t(self.theta), avg_t(self.y)

    def _rho(self, a, x):
        mu = self.policy(x, self.getWeights())
        return 1.0/mu[a] if np.argmax(self.theta.dot(x)) == a else 0.0

    def applyUpdate(self, x, a, xp, r, gamma):
        # Default grads = GTD2

        rho = self._rho(a,x)
        self.z *= gamma*self.lmda*rho
        self.z[a] += x

        dw, dh = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw, dh)

        # extra-gradient step
        rho = self._rho(a,x)
        self.z *= gamma*self.lmda*rho
        self.z[a] += x

        dw, dh = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw, dh)

        self.av_theta.update(self.theta); self.av_y.update(self.y)

    def _apply(self, dw, dh):
        self.theta[a] += self.alpha * dw
        self.y[a] += self.alpha * dh

    def grads(self, x, a, xp, r, gamma, rho):
        q_a = self.theta[a].dot(x)
        qp_m = self.theta.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a

        dw = rho * self.z[a].dot(self.y[a])*(x-gamma*xp)
        dh = rho*delta*self.z[a] - x.dot(self.y[a])*x
        return dw, dh

    def getWeights(self):
        return self.av_theta.get()

    def initWeights(self, u):
        self.theta = u
        self.av_theta.reset(np.array(u))
