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
        u = np.random.rand(features)
        normu = norm(u)
        self.u = u if normu <=1 else u/normu

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
        self.u = u if unorm <= 1 else u/unorm
        self.W = unorm
        self.beta = 0.0

class ParamUntrunc(Param):
    '''
    Parameter-free OLO algorithm with gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta: float):
        super().__init__(features, W0, g, beta)

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
        self.G += norm(g)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * g
        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm
