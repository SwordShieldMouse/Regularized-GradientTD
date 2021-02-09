import numpy as np

from src.agents.TD import TD

class Vtrace(TD):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

    def grads(self, x, a, xp, r, gamma, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + gamma * vp - v

        rho_hat = np.min((rho, 1))
        return rho_hat * delta * x

    def update(self, x, a, xp, r, gamma, rho):
        dw = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw)

    def _apply(self, dw):
        self.w = self.proj(self.w + self.alpha * dw)

    def initWeights(self, u):
        self.w = u

    def getWeights(self):
        return self.w
