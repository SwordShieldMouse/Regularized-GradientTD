import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.agents.BaseAgent import BaseAgent

class ESARSA(BaseAgent):
    def applyUpdate(self, x, a, xp, r, gamma):
        raise Exception('on-policy methods no longer work. merp merp')
        ap = self.selectAction(xp)
        q_a = self.w[a].dot(x)

        pi = self.policy(x)
        eqp = self.w.dot(xp).dot(pi)

        g = r + gamma * eqp
        delta = g - q_a

        dw = delta * x

        self.w[a] += self.alpha * dw
        return ap, delta
