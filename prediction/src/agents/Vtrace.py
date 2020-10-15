import numpy as np

from src.agents.BaseAgent import BaseAgent

class Vtrace(BaseAgent):
    def __init__(self, features, params):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']

        self.w = np.zeros(features)

    def update(self, x, a, r, xp, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + self.gamma * vp - v

        rho_hat = np.min((rho, 1))
        self.w = self.w + self.alpha * rho_hat * delta * x

    def initWeights(self, u):
        self.w = u

    def getWeights(self):
        return self.w
