import numpy as np
from numpy.linalg import norm

class CWParam:
    def __init__(self, features: int, W0: float, g: float, beta:float):
        self.beta = beta * np.ones(features)
        self.W = W0 * np.ones(features)
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
            np.minimum(self.beta - 2.0 * np.divide(m, (2.0-np.log2(3.0))*self.A + self.eps), 0.5 / (self.h + self.eps)),
            -0.5 / (self.h + self.eps)
        )

        # update wealth
        self.W -= np.multiply(gtrunc,self.x)

class CWPFGTD:
    """
    Coordinate-wise Parameter-free GTD with hints
    """
    def __init__(self, features, params):
        self.features = features
        self.params = params
        self.gamma = params['gamma']

        # opt params
        self.theta = CWParam(features, params["wealth"], params["hint"], params["beta"])
        self.y = CWParam(features, params["wealth"], params["hint"], params["beta"])

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
        self.theta.W = 2*u
        self.theta.beta = 0.5
        self.av_theta = self.theta.bet()
