import numpy as np

from src.agents.BaseAgent import BaseAgent

class TD(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.alpha = params['alpha']
        self.w = np.zeros(features)

        # for batch_update() to work efficiently, want to
        # stack the param vectors and thus need their dimension.
        # TD only needs 1 param vector, whereas generally the
        # gradient methods need 2
        self.paramShape = (1, self.features)

        self.z = np.zeros(features)
        self.lmda = params.get('lambda', 0.0)

    def grads(self, x, a, xp, r, gamma, rho):
        self.z = rho*(self.z * gamma * self.lmda + x)

        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + gamma * vp - v
        return rho * delta * self.z

    def update(self, x, a, xp, r, gamma, rho):
        dw = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw)

    def _apply(self, dw):
        self.w = self.w + self.alpha * dw

    def initWeights(self, u):
        self.w = u

    def getWeights(self):
        return self.w
