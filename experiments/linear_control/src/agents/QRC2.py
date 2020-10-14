import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class QRC2(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.beta = params.get('beta', 1)
        self.h = np.zeros(features)
        self.last_w = self.w

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
        return sample(self.policy(x, self.w))

    def applyUpdate(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        mu = self.policy(x, self.last_w)
        pi = self.policy(x, self.w)
        p = pi[a] / mu[a]

        q_a = self.w[a].dot(x)

        qp_m = self.w.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a

        delta_hat = self.h.dot(x)

        dw = delta * x - gamma * delta_hat * xp
        dh = p * (delta - delta_hat) * x - self.beta * self.h

        self.last_w = self.w.copy()

        self.h += self.alpha * dh
        self.w[a] += self.alpha * dw

        return ap, delta
