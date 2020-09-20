import numpy as np

class PFGTDH:
    """
    Parameter-free GTD with hints
    """
    def __init__(self, features, params):
        self.features = features
        self.params = params
        
        self.gamma = params['gamma']

        self.lower_bound = -1e5
        self.upper_bound = 1e5
        self.wealth = params['wealth'] # has to be more than 0 and of dimension 2d
        self.hints = np.ones(features * 2)
        self.beta = np.zeros(features * 2) # betting fraction
        self.theta = np.zeros(features) # primal variable
        self.y = np.zeros(features) # dual variable
        self.G = np.zeros(features * 2)
        self.u = np.random.uniform(size = 2 * features) # initial direction

        # average weights
        self.avg_theta = np.zeros(features)
        self.avg_y = np.zeros(features)

        # keep track of time
        self.t = 0

        ## for numerical stability
        self.eps = 1e-8

        ## stuff for ONS
        self.A = np.zeros(features * 2)

    def update(self, x, a, r, xp, rho):

        # assume g = (g_\theta, -g_y)
        # choose point via coin-betting
        u = np.concatenate((self.theta, self.y))
        v = self.beta * self.wealth
        w = v * u

        # project if there are constraints
        z = np.clip(w, self.lower_bound, self.upper_bound)

        # otherwise, don't project
        # z = w

        theta = z[ : self.features]
        y = z[self.features : ]

        # construct the gradient
        M = np.outer(x, x)
        b = rho * r * x
        A = rho * np.outer(x, x - self.gamma * xp)
        g = np.concatenate((-A.T @ y, A @ theta + M @ y - b))

        # constrained reduction to unconstrained
        g_norm = np.abs(g)
        S_Z_grad = np.zeros(2 * self.features)
        for i in range(2 * self.features):
            if w[i] < self.lower_bound:
                S_Z_grad[i] = w[i] - self.lower_bound
            elif w[i] > self.upper_bound:
                S_Z_grad[i] = w[i] - self.upper_bound
        gtilde = 1 / 2 * g + g_norm * S_Z_grad
        gtilde_norm = np.abs(gtilde)
        ineq_ind = (gtilde_norm > self.hints).astype(int)
        gtrunc = ineq_ind * (self.hints * gtilde / (gtilde_norm + self.eps)) + (1 - ineq_ind) * gtilde # added a numerical constant for stability
        assert np.isnan(gtrunc).sum() == 0, "nans in gtrunc"
        self.hints = np.maximum(self.hints, gtilde_norm)
        
        # update betting parameters
        s = gtrunc @ u
        m = s / (1 - self.beta * s)
        self.A += np.power(m, 2)
        beta_minarg1 = self.beta - 2 * m / ((2 - np.log(3)) * self.A + self.eps)
        beta_minarg2 = 1 / ( 2 * self.hints + self.eps)
        beta_maxarg1 = np.minimum(beta_minarg1, beta_minarg2)
        beta_maxarg2 = -1 / ( 2 * self.hints + self.eps)
        self.beta = np.maximum(beta_maxarg1, beta_maxarg2)
        assert np.isnan(self.beta).sum() == 0, "nans in beta"
        self.wealth -= s * v

        # update the weights
        self.G += gtilde_norm ** 2
        next_u = self.u - np.sqrt(2) * gtrunc / (2 * np.sqrt(self.G) + self.eps)
        self.u = next_u / np.abs(next_u)
        self.theta, self.y = (next_u[ : self.features], next_u[self.features : ])
        assert np.isnan(self.theta).sum() == 0, "nans in theta"
        assert np.isnan(self.y).sum() == 0, "nans in y"

        # update averages
        self.t += 1
        self.avg_theta = self.avg_theta * (self.t - 1) / self.t + self.theta / self.t
        self.avg_y = self.avg_y * (self.t - 1) / self.t + self.y / self.t

        

    def getWeights(self):
        return self.avg_theta
