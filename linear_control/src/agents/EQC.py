import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class EQC(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.h = np.zeros((actions, features))

    def applyUpdate(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        pi = self.policy(x)
        eqp = self.w.dot(xp).dot(pi)

        g = r + gamma * eqp
        delta = g - q_a

        delta_hat = self.h.dot(x).dot(pi)

        dw = delta * x - gamma * delta_hat * xp

        self.h[a] += self.alpha * (delta - delta_hat) * x
        self.w[a] += self.alpha * dw
        return ap, delta
