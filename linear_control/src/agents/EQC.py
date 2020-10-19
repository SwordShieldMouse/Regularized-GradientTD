import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class EQC(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.w = np.zeros((actions, features))
        self.h = np.zeros((actions, features))

    def grads(self, x, a, xp, r, gamma, rho):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        pi = self.policy(x)
        eqp = self.w.dot(xp).dot(pi)

        g = r + gamma * eqp
        delta = g - q_a
        delta_hat = self.h.dot(x).dot(pi)

        dw = delta * x - gamma * delta_hat * xp
        dh = (delta-delta_hat)*x
        return dw, dh

    def _apply(self, dw, dh, a):
            self.h[a] += self.alpha * dh
            self.w[a] += self.alpha * dw


    def applyUpdate(self, x, a, xp, r, gamma):
        dw, dh = self.grads(x, a, xp, r, gamma, 1.0)
        self._apply(dw, dh, a)
        return None, None
