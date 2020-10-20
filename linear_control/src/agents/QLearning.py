import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class QLearning(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.w = np.zeros((actions, features))
        self.paramShape = (1, actions, features)

    def grads(self, x, a, xp, r, gamma, rho):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        qp = self.w.dot(xp).max()

        g = r + gamma * qp
        delta = g - q_a

        dw = delta * x
        return dw

    def _apply(self, dw, a):
        self.w[a] += self.alpha * dw

    def applyUpdate(self, x, a, xp, r, gamma):
        dw = self.grads(x, a, xp, r, gamma, 1.0)
        self._apply(dw, a)
        return None, None

    def batch_update(self, gen, num):
        exps = gen.sample(samples=num)
        for i in range(num):
            grad = self.grads(*exps[i])
            self._apply(grad, exps[i][1])
