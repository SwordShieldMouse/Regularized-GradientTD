import numpy as np
from numpy.linalg import norm

'''
Parameter-free OLO algorithms used to construct
parameter-free policy evaluation algs
'''
class Param:
    '''
    Parameter-free OLO algorithm with gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta: float):
        self.beta = beta
        self.W = W0
        self.h = g

        # initial bet
        self.v = self.beta * self.W

        # random initial direction and normalize
        u = np.ones(features)
        normu = norm(u)
        self.u = u if normu<=1 else u/normu

        self.A = 0.0
        self.G = 0.0

        self.eps = 1e-5

        # NOTE: unused for now
        self.lower_bound = np.finfo(np.float64).min / 1e150
        self.upper_bound = np.finfo(np.float64).max / 1e150

    def bet(self):
        self.v = self.beta * self.W
        return self.v * self.u

    def update(self, g):
        # NOTE: have completely removed the constraint set for now

        # Incorporate grad bound
        gradnorm = norm(g)
        gtrunc = g if gradnorm < self.h else self.h*g / (gradnorm + self.eps)
        self.h = max(self.h, gradnorm)

        # update betting fraction
        s = np.dot(gtrunc, self.u)
        m = s / (1.0 - self.beta * s)
        self.A += m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= s*self.v

        # update directional weights
        self.G += norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

    def initWeights(self, u):
        unorm = norm(u)
        self.u = u if unorm <=1 else u/unorm
        # arbitrary; bet 1/10 of wealth if we have to
        # be initialized to something other than 0
        self.W = 2.0*unorm
        self.beta = 0.5

class VectorHintsParam:
    '''
    Parameter-free OLO algorithm with gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta: float):
        self.beta = beta
        self.W = W0 * features
        self.vec_h = np.ones(features)*g/np.sqrt(features)
        self.h = norm(self.vec_h)

        # initial bet
        self.v = self.beta * self.W

        # random initial direction and normalize
        u = np.ones(features)
        normu = norm(u)
        self.u = u if normu<=1 else u/normu

        self.A = 0.0
        self.G = 0.0

        self.eps = 1e-5

        # NOTE: unused for now
        self.lower_bound = np.finfo(np.float64).min / 1e150
        self.upper_bound = np.finfo(np.float64).max / 1e150

    def bet(self):
        self.v = self.beta * self.W
        return self.v * self.u

    def update(self, g):
        # NOTE: have completely removed the constraint set for now

        # Incorporate grad bound
        gradnorm = np.abs(g)

        # Truncate g coordinate-wise with a vector of hints
        gtrunc = g.copy()
        truncIdx = np.argwhere(gradnorm > self.vec_h)
        gtrunc[truncIdx] = np.multiply(self.vec_h[truncIdx], g[truncIdx]) / (gradnorm[truncIdx] + self.eps)
        self.vec_h = np.maximum(self.vec_h, gradnorm)

        # pass norm of vector hint
        self.h = norm(self.vec_h)

        # update betting fraction
        s = np.dot(gtrunc, self.u)
        m = s / (1.0 - self.beta * s)
        self.A += m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= s*self.v

        # update directional weights
        self.G += norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

    def initWeights(self, u):
        unorm = norm(u)
        self.u = u if unorm <=1 else u/unorm
        self.W = unorm*2.0
        self.beta = 0.5

class CWParam:
    '''
    Coordinate-wise parameter-free OLO algorithm with
    gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.beta = beta * np.ones(features)
        self.W = np.ones(features) * W0
        self.h = g * np.ones(features)

        # initial bet
        self.x = np.multiply(self.beta, self.W)

        self.A = np.zeros(features)

        self.eps = 1e-8

        # NOTE: unused for now
        self.lower_bound = np.finfo(np.float64).min / 1e150
        self.upper_bound = np.finfo(np.float64).max / 1e150

    def bet(self):
        self.x = np.multiply(self.beta, self.W)
        return self.x

    def update(self, g):
        # NOTE: have completely removed the constraint set for now

        # Incorporate grad bound
        gradnorm = np.abs(g)

        gtrunc = g.copy()
        truncIdx = np.argwhere(gradnorm > self.h)
        gtrunc[truncIdx] = np.multiply(self.h[truncIdx], g[truncIdx]) / (gradnorm[truncIdx] + self.eps)
        self.h = np.maximum(self.h, gradnorm)

        # update betting fraction
        m = np.divide(gtrunc,  1.0 - np.multiply(self.beta, gtrunc))
        self.A += np.power(m,2)
        self.beta = np.maximum(
            np.minimum(self.beta - 2.0 * np.divide(m, (2.0-np.log(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= np.multiply(gtrunc,self.x)

    def initWeights(self, u):
        assert u.shape == self.W.shape
        self.W = u*2.0
        self.beta = 0.5

