import numpy as np
import numpy.linalg as LA

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
        self.wealth = np.array(params['wealth'])
        self.hints = np.ones(2)
        self.beta = np.zeros(2) # betting fraction, one for each of primal and dual
        self.theta = np.zeros(features) # primal variable
        self.y = np.zeros(features) # dual variable
        self.G = np.zeros((features, 2))
        self.u = np.random.uniform(size = (features, 2)) # initial direction for each of primal and dual
        self.u /= self.norm(self.u)
        self.A = np.zeros(2)

        # average weights
        self.avg_theta = np.zeros(features)
        self.avg_y = np.zeros(features)

        # keep track of time
        self.t = 0

        ## for numerical stability
        self.eps = 1e-8

    def norm(self, g):
        return LA.norm(g, axis = 0, keepdims = True)


    def update(self, x, a, r, xp, rho):
        # do updates separately on theta and y
        # assume g = (g_\theta, -g_y)
        # choose point via coin-betting
        v = self.beta * self.wealth
        # print(v.shape, self.beta.shape, self.wealth.shape)
        w = self.u * v.reshape((1, 2))
        assert w.shape == (self.features, 2), w.shape

        # project if there are constraints
        z = np.clip(w, self.lower_bound, self.upper_bound)

        # otherwise, don't project
        # z = w

        theta = z[:, 0]
        y = z[:, 1]
        assert np.isnan(theta).sum() == 0, "nans in theta"
        assert np.isnan(y).sum() == 0, "nans in y"

        # construct the gradient
        M = np.outer(x, x)
        b = rho * r * x
        A = rho * np.outer(x, x - self.gamma * xp)
        g_theta = -A.T @ y
        g_y = A @ theta + M @ y - b
        g = np.stack((g_theta, g_y)).transpose((1, 0))
        assert g.shape == (self.features, 2)

        # constrained reduction to unconstrained
        S_Z_grad = np.zeros((self.features, 2))
        for i in range(self.features):
            for j in range(2):
                if w[i, j] < self.lower_bound:
                    S_Z_grad[i, j] = w[i, j] - self.lower_bound
                elif w[i, j] > self.upper_bound:
                    S_Z_grad[i, j] = w[i, j] - self.upper_bound
        gtilde = 1 / 2 * g + self.norm(g) * S_Z_grad
        ineq_ind = (self.norm(gtilde) > self.hints).astype(int)
        gtrunc = ineq_ind * (self.hints * gtilde / (self.norm(gtilde) + self.eps)) + (1 - ineq_ind) * gtilde # added a numerical constant for stability
        assert np.isnan(gtrunc).sum() == 0, f"nans in gtrunc, {gtrunc}"
        self.hints = np.maximum(self.hints, self.norm(gtilde).squeeze())
        
        # update betting parameters
        # print(gtrunc.shape, self.u.shape)
        s = np.sum(gtrunc * self.u, axis = 0)
        m = s / (1 - self.beta * s)
        self.A += np.power(m, 2)
        beta_minarg1 = self.beta - 2 * m / ((2 - np.log(3)) * self.A + self.eps)
        beta_minarg2 = 1 / ( 2 * self.hints + self.eps)
        beta_maxarg1 = np.minimum(beta_minarg1, beta_minarg2)
        beta_maxarg2 = -1 / ( 2 * self.hints + self.eps)
        self.beta = np.maximum(beta_maxarg1, beta_maxarg2)
        assert np.isnan(self.beta).sum() == 0, "nans in beta"
        self.wealth -= s * v
        # print(self.beta.shape)

        # update the weights
        self.G += self.norm(gtrunc) ** 2
        next_u = self.u - np.sqrt(2) * gtrunc / (2 * np.sqrt(self.G) + self.eps)
        self.u = next_u / LA.norm(next_u)

        # update averages
        self.t += 1
        self.avg_theta = self.avg_theta * (self.t - 1) / self.t + theta / self.t
        self.avg_y = self.avg_y * (self.t - 1) / self.t + y / self.t

        

    def getWeights(self):
        return self.avg_theta
