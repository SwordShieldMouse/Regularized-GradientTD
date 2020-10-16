import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class GQ(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.theta = np.zeros((actions, features))
        self.y = np.zeros((actions, features))

        avg_t = getattr(Averages, params.get('averaging', "Uniform"))
        self.av_theta, self.av_y = avg_t(self.theta), avg_t(self.y)

        self.lmda = params.get('lambda', 0.0)
        self.z = np.zeros(features)

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

    def _rho(self, a, x):
        mu = self.policy(x, self.getWeights())
        return 1.0/mu[a] if np.argmax(self.theta_t.dot(x)) == a else 0.0

    def applyUpdate(self, x, a, xp, r, gamma):
        # Default grads = GTD2

        rho = self._rho(a, x)
        self.z = self.z * gamma * self.lmda * rho + x

        dw, dh = self.grads(x, a, xp, r, gamma, rho, self.theta, self.y)
        self.theta[a] += self.alpha * dw
        self.y[a] += self.alpha * dh

        # extra-gradient step
        dw, dh = self.grads(x, a, xp, r, gamma, rho, self.theta, self.y)
        self.theta[a] += self.alpha * dw
        self.y[a] += self.alpha * dh

        self.av_theta.update(self.theta); self.av_y.update(self.y)

    def grads(self, x, a, xp, r, gamma, rho, theta, y):
        q_a = theta.dot(x)
        qp_m = theta.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a


        dw = rho * self.z.dot(y[a])*(x-gamma*xp)
        dh = rho*delta*self.z - x.dot(y[a])*x
        return dw, dh

    def getWeights(self):
        return self.av_theta.get()

    def initWeights(self, u):
        self.theta = u
        self.av_theta.reset(np.array(u))
