import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class QLearning(BaseAgent):
    def applyUpdate(self, x, a, xp, r, gamma):
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        qp = self.w.dot(xp).max()

        g = r + gamma * qp
        delta = g - q_a

        dw = delta * x

        self.w[a] += self.alpha * dw
        return ap, delta
