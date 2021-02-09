import numpy as np
from numpy.linalg import norm

class BaseAgent:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # used in batch_update
        self.paramShape = (2, self.features)

        self.D = params.get('D', np.inf)

    def batch_update(self, gen, num):
        exps = gen.sample(samples=num)
        for i in range(num):
            self.update(*exps[i])

    def grads(self, x, a, xp, r, gamma, rho):
        raise(NotImplementedError("Agent.grads not implemented"))

    def _apply(self, g):
        '''Update weights given a grad'''
        raise(NotImplementedError('Agent._apply not implemented'))

    def value(self, X):
        return np.dot(X, self.getWeights())

    def getWeights(self):
        raise(NotImplementedError("getWeights not implemented"))

    def proj(self, x):
        normx = norm(x)
        return x if normx<=self.D else (x/normx)*self.D
