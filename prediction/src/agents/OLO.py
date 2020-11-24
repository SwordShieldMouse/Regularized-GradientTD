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
        self.W = 1.0 if unorm <= 1.0 else unorm
        self.beta = 1.0

class HalfCWParam:
    '''
    Parameter-free OLO algorithm with gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta: float):
        self.beta = beta
        self.W = W0 / features
        self.h = g

        # initial bet
        self.v = self.beta * self.W

        # random initial direction and normalize
        u = np.ones(features)
        normu = norm(u)
        self.u = u if normu<=1 else u/normu

        self.A = 0.0
        self.G = np.zeros(features)

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
        self.G += np.power(gtrunc, 2)
        u = self.u - np.sqrt(2)*np.divide(gtrunc, np.sqrt(self.G) + self.eps)

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

    def initWeights(self, u):
        unorm = norm(u)
        self.u = u if unorm <=1 else u/unorm
        self.W = 1.0 if unorm <= 1.0 else unorm
        self.beta = 1.0

'''
Parameter-free OLO algorithms used to construct
parameter-free policy evaluation algs
'''
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

class SCParam:
    '''
    Parameter-free OLO algorithm with gradient bound hints and curvature adaptation.
    Improves to linear convergence when the losses are strongly-convex
    '''
    def __init__(self, features: int, W0: float, g: float, beta:float):
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

        self.av_bet = np.zeros(features)
        self.av_n = 1.0

        self.eps = 1e-5

        # NOTE: unused for now
        self.lower_bound = np.finfo(np.float64).min / 1e150
        self.upper_bound = np.finfo(np.float64).max / 1e150

    def bet(self):
        self.v = self.beta * self.W
        self.z = self.v * self.u + self.av_bet/self.av_n
        return self.z

    def update(self, g):
        # NOTE: have completely removed the constraint set for now

        gradnorm = norm(g)
        self.av_bet += norm(g)**2 * self.z
        self.av_n += norm(g)**2

        # Incorporate grad bound
        gtrunc = g if gradnorm < self.h else self.h*g / (gradnorm + self.eps)
        self.h = max(self.h, gradnorm)

        # update betting fraction
        s = np.dot(gtrunc, self.u)
        m = s / (1.0 - self.beta * s)
        self.A += m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / self.h),
            -0.5 / self.h
        )

        # update wealth
        self.W -= s*self.v

        # update directional weights
        self.G += norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

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
        self.W = u
        self.beta = 1.0

class CWParamScalarHint:
    '''
    Coordinate-wise parameter-free OLO algorithm with
    gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.beta = beta * np.ones(features)
        self.W = np.ones(features) * W0
        self.h = g

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
        gradnorm = norm(g)
        gtrunc = g if gradnorm < self.h else self.h*g / (gradnorm + self.eps)
        self.h = max(self.h, gradnorm)

        # update betting fraction
        m = np.divide(gtrunc,  1.0 - np.multiply(self.beta, gtrunc))
        self.A += np.power(m,2)
        self.beta = np.maximum(
            np.minimum(self.beta - 2.0 * np.divide(m, (2.0-np.log(3.0))*self.A + self.eps), 0.5 / self.h),
            -0.5 / self.h
        )

        # update wealth
        self.W -= np.multiply(gtrunc,self.x)

    def initWeights(self, u):
        assert u.shape == self.W.shape
        self.W = u
        self.beta = 1.0

class DiscountedParam:
    '''
    Parameter-free OLO algorithm with gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta: float, gamma: float):
        self.beta = beta
        self.W = W0
        self.h = g

        # initial bet
        self.v = self.beta * self.W

        # random initial direction and normalize
        u = np.ones(features)
        normu = norm(u)
        self.u = u if norm(u) <= 1 else u/normu

        self.A = 0.0
        self.G = 0.0
        self.gamma = gamma

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
        self.A = self.A + m**2
        self.beta = max(
            min(self.beta - 2.0*m / ((2.0-np.log(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= s*self.v

        # update directional weights
        self.G = (1.0-self.gamma) * self.G + self.gamma * norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

    def initWeights(self, u):
        unorm = norm(u)
        self.u = u if unorm <=1 else u/unorm
        self.W = 1.0 if unorm <= 1.0 else unorm
        self.beta = 1.0

class COCOBParam:
    '''
    Coordinate-wise parameter-free OLO algorithm with
    gradient-bound hints
    '''
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.w = np.zeros(features)
        self.w1 = self.w.copy()
        self.h = W0 * np.ones(features)

        self.reward = np.zeros(features)
        self.theta = np.zeros(features)
        self.G = self.h.copy()

        self.eps = 1e-8

    def bet(self):
        return self.w

    def sigma(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def update(self, g):
        # NOTE: have completely removed the constraint set for now
        g *=-1

        # Incorporate grad bound
        gradnorm = np.abs(g)

        gtrunc = g.copy()
        truncIdx = np.argwhere(gradnorm > self.h)
        gtrunc[truncIdx] = np.multiply(self.h[truncIdx], g[truncIdx]) / (gradnorm[truncIdx] + self.eps)
        self.h = np.maximum(self.h, gradnorm)

        # update betting fraction
        self.G += np.abs(gtrunc)
        self.reward += (self.w - self.w1) * gtrunc
        self.theta += gtrunc
        self.beta = np.divide(2*self.sigma(2*self.theta / (self.G + self.h))-1, self.h)
        self.w = self.w1 + self.beta * (self.h + self.reward)

    def initWeights(self, u):
        self.w = u
        self.w1 = self.w.copy()
