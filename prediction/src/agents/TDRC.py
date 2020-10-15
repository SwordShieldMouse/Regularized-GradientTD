import numpy as np

from src.agents.BaseAgent import BaseAgent

class TDRC(BaseAgent):
    def __init__(self, features, params):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.eta = params.get('eta', 1)

        self.w = np.zeros(features)
        self.h = np.zeros(features)

    def update(self, x, a, r, xp, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + self.gamma * vp - v
        delta_hat = self.h.dot(x)

        dw = rho * (delta * x - self.gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x - self.beta * self.h

        self.w = self.w + self.alpha * dw
        self.h = self.h + self.eta * self.alpha * dh

    def getWeights(self):
        return self.w

    def initWeights(self, u):
        self.w = u
