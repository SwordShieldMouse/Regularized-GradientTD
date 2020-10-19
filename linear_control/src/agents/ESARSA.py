import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class ESARSA(BaseAgent):
    def __init__(features, actions, params):
        super().__init__(feature, actions, params)
        self.w = np.zeros(actions, features)
        self.paramShape = (1,actions, features)

    def grads(self, x, a, xp, r, gamma, rho):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        pi = self.policy(x)
        eqp = self.w.dot(xp).dot(pi)

        g = r + gamma * eqp
        delta = g - q_a

        dw = delta * x
        return dw

    def _apply(self, dw, a):
        self.w[a] += self.alpha * dw

    def applyUpdate(self, x, a, xp, r, gamma):
        dw = self.grads(x, a, xp, r, gamma, 1.0)
        self._apply(dw, a)
        return None, None
