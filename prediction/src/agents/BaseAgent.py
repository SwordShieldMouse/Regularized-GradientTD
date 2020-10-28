import numpy as np

class BaseAgent:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # used in batch_update
        self.paramShape = (2, self.features)

    def batch_update(self, gen):
        num = self.params['batch_size']
        exps = gen.sample(samples=num)
        grads = np.zeros(shape)
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
