import numpy as np
from numpy.linalg import norm

class Param:
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.beta = beta
        self.W = W0
        self.h = g

        # initial bet
        self.v = self.beta * self.W
       
        # random initial direction and normalize
        self.u = 2 * np.random.rand(features) - 1.0
        self.u /= norm(self.u)

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
            min(self.beta - 2.0*m / ((2.0-np.log2(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= s*self.v

        # update directional weights
        self.G += norm(gtrunc)**2
        u = self.u - np.sqrt(2)/(2*np.sqrt(self.G) + self.eps) * gtrunc

        unorm = norm(u)
        self.u = u if unorm<=1 else u / unorm

class PFGTDH:
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features, params):
        self.features = features
        self.params = params
        self.gamma = params['gamma']

        # opt params
        self.theta = Param(features, params["wealth"], params["hint"], params["beta"])
        self.y = Param(features, params["wealth"], params["hint"], params["beta"])

        # Average decisions
        self.av_theta = np.zeros(features)
        self.av_y = np.zeros(features)

        self.t = 0.0

    def update(self, x, a, r, xp, rho):
        self.t +=1

        # get bets
        theta_t = self.theta.bet()
        y_t = self.y.bet()

        self.av_theta += 1.0 / self.t * (theta_t - self.av_theta)
        self.av_y += 1.0 / self.t * (y_t - self.av_y)

        # construct gradients
        #
        # ================================
        # --- EFFICIENT IMPLEMENTATION ---
        # ================================
        #  implicitly compute A to avoid
        #  the outerproduct op
        # --------------------------------
        d = x - self.gamma * xp
        g_theta = - rho * d * np.dot(x, y_t)
        g_y = rho * x * np.dot(d, theta_t) - rho * r * x + x * np.dot(x, y_t)

        # ===========================
        # --- Slow implementation ---
        # ===========================
        # useful for debugging
        # ---------------------------
        # d = x - self.gamma * xp
        # At = rho * np.outer(x, d)
        # bt = rho* r * x
        # Mt = np.outer(x, x)
        #
        # g_theta = np.matmul(- At.transpose(), y_t)
        # g_y = np.matmul(At, theta_t) - bt + np.matmul(Mt, y_t)

        self.theta.update(g_theta)
        self.y.update(g_y)

    def getWeights(self):
        return self.av_theta

    def setInitialBet(self, u):
        unorm = norm(u)
        self.theta.u = u/unorm
        self.theta.W = unorm * 2
        self.theta.beta = 0.5
        self.av_theta = self.theta.bet()
