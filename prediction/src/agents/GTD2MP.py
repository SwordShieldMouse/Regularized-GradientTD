import numpy as np

from src.agents.GTD2 import GTD2

class GTD2MP(GTD2):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.alpha = params['alpha']
        self.eta = params.get('eta', 1)

        self.theta = np.zeros(features)
        self.y = np.zeros(features)

        self.av_theta = np.zeros(features)
        self.alpha_1_t = 0.0
        self.t = 0.0

    def update(self, x, a, xp, r, gamma, rho):
        self.t += 1.0

        dtheta, dy = self.grads(x ,a, xp, r, gamma, rho)
        self._apply(dtheta, dy)

        # extra-gradient step
        dtheta, dy = self.grads(x ,a, xp, r, gamma, rho)
        self._apply(dtheta, dy)

        self.alpha_1_t+= self.alpha
        ss = self.alpha / self.alpha_1_t
        self.av_theta = ss * self.theta + (1.0-ss) * self.av_theta

    def getWeights(self):
        return self.av_theta

    def initWeights(self, u):
        self.theta = self.proj(u)
        self.av_theta = u
