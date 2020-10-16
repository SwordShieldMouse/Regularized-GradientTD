import numpy as np

class BaseAgent:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

    def update(self, x, a, xp, r, gamma, rho):
        raise(NotImplementedError('Agent.update not implemented'))

    def value(self, X):
        return np.dot(X, self.getWeights())

    def getWeights(self):
        raise(NotImplementedError("getWeights not implemented"))
