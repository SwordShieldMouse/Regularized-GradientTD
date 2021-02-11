import numpy as np

from src.agents.BaseAgent import BaseAgent

class TDRC(BaseAgent):
    def __init__(self, features, actions, params):
        super().__init__(features, actions, params)

        self.alpha = params['alpha']
        self.beta = params['beta']
        self.eta = params.get('eta', 1)

        self.w = np.zeros(features)
        self.h = np.zeros(features)

    def grads(self, x, a, xp, r, gamma, rho):
        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + gamma * vp - v
        delta_hat = self.h.dot(x)

        dw = rho * (delta * x - gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x - self.beta * self.h
        return dw, dh

    def update(self, x, a, xp, r, gamma, rho):
        dw, dh = self.grads(x, a, xp, r, gamma, rho)
        self._apply(dw,dh)

    def _apply(self, dw, dh):
        self.w = self.proj(self.w + self.alpha * dw)
        self.h = self.proj(self.h + self.eta * self.alpha * dh)

    def getWeights(self):
        return self.w

    def initWeights(self, u):
        self.w = self.proj(u)

class BatchTDRC(TDRC):
    def __init__(self, features, actions, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, actions, params)
        self.t = 0.0
        self.av_w = np.zeros(features)

    def update(self, x, a, xp, r, gamma, rho):
        self.t+=1.0
        super().update(x, a, xp, r, gamma, rho)
        self.av_w += 1.0 / self.t * (self.w - self.av_w)

    def initWeights(self, u):
        self.w = self.proj(u)
        self.av_w = self.w.copy()

    def getWeights(self):
        return self.av_w
