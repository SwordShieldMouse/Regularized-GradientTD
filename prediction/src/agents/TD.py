import numpy as np

from src.agents.BaseAgent import BaseAgent

class TD(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.alpha = params['alpha']
        self.w = np.zeros(features)

    def grads(self, x, a, xp, r, gamma, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + gamma * vp - v
        return rho * delta * x, None

    def update(self, x, a, xp, r, gamma, rho):
        dw, dh = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw, dh)

    def _apply(self, dw, dh):
        self.w = self.w + self.alpha * dw

    def initWeights(self, u):
        self.w = u

    def getWeights(self):
        return self.w
