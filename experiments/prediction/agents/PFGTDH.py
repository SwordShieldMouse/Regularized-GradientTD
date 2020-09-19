import numpy as np

class FPGTDH():
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features, params):
        self.features = features
        self.params = params
        
        self.gamma = params['gamma']
        self.bounds = params["bounds"] # of dim (2d, 2)

        self.wealth = params['wealth'] # has to be more than 0 and of dimension 2d
        self.hints = np.ones(features * 2)
        self.beta = np.zeros(features * 2) # betting fraction
        self.theta = np.zeros(features) # primal variable
        self.y = np.zeros(features) # dual variable
        self.G = np.zeros(features * 2)

        ## stuff for ONS
        self.A = np.zeros(features * 2)

    def update(self, x, a, r, xp, rho):

        # assume g = (g_\theta, -g_y)
        # choose point via coin-betting
        u = np.concat((self.theta, self.y))
        v = self.beta * self.wealth
        w = v * u
        theta = w[ : self.features]
        y = w[self.features : ]

        # construct the gradient
        M = np.outer(x, x)
        b = rho * r * x
        A = rho * np.outer(x, x - self.gamma * xp)
        g = np.concatenate((-A.T @ y, A @ theta + M @ y - b)) # of dim 2d, one gradient for each param

        # ignore constraints lol
        gtilde = 1 / 2 * g
        gtilde_norm = np.abs(gtilde)
        gtrunc = (gtilde_norm > self.hints) * (self.hints * gtilde / gtilde_norm) + (gtilde_norm <= self.hints) * gtilde
        self.hints = np.max(self.hints, gtilde_norm)
        
        # update betting parameters
        s = gtrunc @ u
        m = s / (1 - self.beta * s)
        self.A += np.power(m, 2)
        self.beta = np.max(np.min(1 / 2 / self.hints, self.beta - 2 * m / ((2 - np.log(3)) * self.A)), -1 / 2 / self.hints)
        self.wealth -= s * w

        # update the weights
        self.G += gtilde_norm ** 2
        # don't bother projecting for the update
        next_u = u - np.sqrt(2) * gtrunc / (2 * np.sqrt(self.G))
        self.theta, self.y = (next_u[ : self.features], next_u[self.features : ])
        

    def getWeights(self):
        return self.theta, self.y
