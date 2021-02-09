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
    def __init__(self, features: int, W0: float, beta: float):
        self.beta = beta
        self.W = W0

        # initial bet
        self.v = self.beta * self.W

        # arbitrary
        u = np.ones(features)
        normu = norm(u)
        self.u = u if normu<=1 else u/normu

        self.A = 0.0
        self.G = 0.0

        self.eps=1e-5

    def bet(self):
        self.v = self.beta * self.W
        return self.v * self.u

    def update(self, gtrunc, h):
        # update betting fraction
        s = np.dot(gtrunc, self.u)
        m = s / (1.0 - self.beta * s)
        self.A += m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / h),
            -0.5 / h
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
    def __init__(self, features: int, W0: float, beta: float):
        self.beta = beta
        self.W = W0 * features

        # initial bet
        self.v = self.beta * self.W

        # random initial direction and normalize
        u = np.ones(features)
        normu = norm(u)
        self.u = u if normu<=1 else u/normu

        self.A = 0.0
        self.G = 0.0

        self.eps = 1e-5

    def bet(self):
        self.v = self.beta * self.W
        return self.v * self.u

    def update(self, gtrunc, vec_h):

        # pass norm of vector hint
        h = norm(vec_h)

        # update betting fraction
        s = np.dot(gtrunc, self.u)
        m = s / (1.0 - self.beta * s)
        self.A += m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / h ),
            -0.5 / h 
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
    def __init__(self, features: int, W0: float, beta:float):
        self.beta = beta * np.ones(features)
        self.W = np.ones(features) * W0

        # initial bet
        self.x = np.multiply(self.beta, self.W)

        self.A = np.zeros(features)

        self.eps = 1e-8

    def bet(self):
        self.x = np.multiply(self.beta, self.W)
        return self.x

    def update(self, gtrunc, vec_h):

        # update betting fraction
        m = np.divide(gtrunc,  1.0 - np.multiply(self.beta, gtrunc))
        self.A += np.power(m,2)
        self.beta = np.maximum(
            np.minimum(self.beta - 2.0 * np.divide(m, (2.0-np.log(3.0))*self.A + self.eps), 0.5 / vec_h),
            -0.5 / vec_h
        )

        # update wealth
        self.W -= np.multiply(gtrunc,self.x)

    def initWeights(self, u):
        assert u.shape == self.W.shape
        self.W = u*2.0
        self.beta = 0.5


class PFPlus:
    def __init__(self, features: int, W0: float, beta:float):
        self.A = VectorHintsParam(features, W0, beta)
        self.B = CWParam(features, W0,  beta)

    def bet(self):
        return self.A.bet()+self.B.bet()

    def initWeights(self, u):
        self.A.initWeights(u/2)
        self.B.initWeights(u/2)

    def update(self, gtrunc, h):
        self.A.update(gtrunc, h)
        self.B.update(gtrunc, h)
