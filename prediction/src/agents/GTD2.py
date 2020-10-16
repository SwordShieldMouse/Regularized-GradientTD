import numpy as np

from src.agents.BaseAgent import BaseAgent

class GTD2(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.alpha = params['alpha']
        self.eta = params.get('eta', 1)

        self.theta = np.zeros(features)
        self.y = np.zeros(features)

    def grads(self, x, a, xp, r, gamma, rho):
        v = self.theta.dot(x)
        vp = self.theta.dot(xp)

        delta = r + gamma * vp - v
        delta_hat = self.y.dot(x)

        dw = rho * (delta_hat * x - gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x
        return dw, dh

    def _apply(self, dtheta, dy):
        # apply the gradient. Used for batch updates
        self.theta = self.theta + self.alpha * dtheta
        self.y = self.y + self.eta * self.alpha * dy

    def update(self, x, a, xp, r, gamma, rho):
        dtheta, dy = self.grads(x, a, xp, r, gamma, rho)
        self._apply_update(dtheta, dy)

    def initWeights(self, u):
        self.theta = u

    def getWeights(self):
        return self.theta

class BatchGTD2(GTD2):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.av_theta = np.zeros(features)
        self.t=0.0

    def update(self, x, a, xp, r, gamma, rho):
        self.t+=1

        dtheta,dy = self.grads(x,a,xp,r,gamma,rho)
        self._apply(dtheta,dy)

        self.av_theta += 1.0/self.t * (self.theta - self.av_theta)

    def initWeights(self, u):
        u = np.array(u, dtype='float64')
        self.theta = u
        self.av_theta = u

    def getWeights(self):
        return self.av_theta
