import numpy as np

class GTD2:
    def __init__(self, features, params):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.eta = params.get('eta', 1)

        self.w = np.zeros(features)
        self.h = np.zeros(features)

        self.avw = np.zeros(features)
        self.t=0.0

    def update(self, x, a, r, xp, rho):
        self.t+=1

        v = self.w.dot(x)
        vp = self.w.dot(xp)

        delta = r + self.gamma * vp - v
        delta_hat = self.h.dot(x)

        dw = rho * (delta_hat * x - self.gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x

        self.w = self.w + self.alpha * dw
        self.h = self.h + self.eta * self.alpha * dh

        self.avw += 1.0/self.t * (self.w - self.avw)

    def getWeights(self):
        return self.avw
