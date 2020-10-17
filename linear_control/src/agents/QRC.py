import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class QRC(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)
        self.beta = params.get('beta', 1)
        self.h = np.zeros((actions, features))

    def applyUpdate(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        qp_m = self.w.dot(xp).max()

        g = r + gamma * qp_m
        delta = g - q_a

        delta_hat = self.h[a].dot(x)

        dw = delta * x - gamma * delta_hat * xp
        dh = (delta - delta_hat) * x - self.beta * self.h[a]

        self.h[a] += self.alpha * dh
        self.w[a] += self.alpha * dw
        return ap, delta
