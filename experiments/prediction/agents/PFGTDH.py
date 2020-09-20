import numpy as np
from numpy.linalg import norm

class Param:
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.beta = beta
        self.W = W0
        self.h = g

        # initial bet
        self.v = self.beta * self.W
       
        # random initial direction
        self.u = 2 * np.random.rand(features) - 1.0
       
        # u should be on the unit sphere
        self.u /= norm(self.u)

        self.A = 0.0
        self.G = 0.0

        self.eps = 1e-8

        # NOTE: unused for now
        self.lower_bound = np.finfo(np.float64).min / 1e150
        self.upper_bound = np.finfo(np.float64).max / 1e150

    def bet(self):
        self.v = self.beta * self.W
        return self.v * self.u

    def update(self, g):
        # NOTE: have completely removed the constraint set for now

        gradnorm = norm(g)
        #print(f"grad: {g}")
        #print(f"gradnorm: {gradnorm}")
        gtrunc = g if gradnorm < self.h else self.h*g / (gradnorm + self.eps)
        #self.h = max(self.h, gradnorm)
        #print(f"h: {self.h}")

        s = np.dot(gtrunc, self.u)
        self.W -= s*self.v

        m = s / (1.0 - self.beta * s)
        self.A += m**2
        print(s)
        print(m)
        print(self.beta)
        print(self.A)
        print(self.beta)
        self.beta = max(min(self.beta - 2.0*m / ((2.0-np.log2(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)), -0.5 / (self.h + self.eps))

        self.G += norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc
        self.u = u if norm(u)<= 1 else u / norm(u)


class PFGTDH:
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features, params):
        self.features = features
        self.params = params


        self.gamma = params['gamma']
        # self.bounds = params["bounds"] # of dim (2d, 2)

        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.av_theta = np.zeros(features)

        self.y = Param(features, params["wealth"], params["hint"], params["beta"])
        self.av_y = np.zeros(features)

        ## for numerical stability
        self.eps = 1e-5
        self.t = 0.0

    def update(self, x, a, r, xp, rho):
        self.t +=1

        # get bets
        theta_t = self.theta.bet()
        y_t = self.y.bet()

        # update averages
        self.av_theta += 1.0 / self.t * (theta_t - self.av_theta)
        self.av_y += 1.0 / self.t * (y_t - self.av_y)

        # construct gradients
        # NOTE: trying to implicitly compute A to avoid the outerproduct op
        d = x - self.gamma * xp
        At = rho * np.outer(x, d)
        bt = rho* r * x
        Mt = np.outer(x, x)
        g_theta = np.matmul(- At.transpose(), y_t)
        g_y = np.dot(np.matmul(At, theta_t) - bt, y_t) + np.matmul(Mt, y_t)

        self.theta.update(g_theta)
        print("---")
        self.y.update(g_y)
        print()

    def getWeights(self):
        return self.av_theta
