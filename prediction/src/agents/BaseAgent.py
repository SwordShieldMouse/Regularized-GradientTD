import numpy as np

class BaseAgent:
    def __init__(self, features, params):
        self.features = features

    def value(self, X):
        return np.dot(X, self.getWeights())

    def getWeights(self):
        raise(NotImplementedError("getWeights not implemented"))
