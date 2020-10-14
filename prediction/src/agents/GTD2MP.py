import numpy as np

class GTD2MP:
    def __init__(self, features, params):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.eta = params.get('eta', 1)

        self.theta = np.zeros(features)
        self.y = np.zeros(features)

        self.thetam = np.zeros(features)
        self.ym = np.zeros(features)

        self.av_theta = np.zeros(features)
        self.alpha_1_tm1 = 0.0
        self.t = 0.0

    def update(self, x, a, r, xp, rho):
        self.t += 1.0

        v = self.theta.dot(x)
        vp = self.theta.dot(xp)

        delta = r + self.gamma * vp - v
        delta_hat = self.y.dot(x)

        dw = rho * (delta_hat * x - self.gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x

        self.thetam = self.theta + self.alpha * dw
        self.ym = self.y + self.eta * self.alpha * dh

        v = self.thetam.dot(x)
        vp = self.thetam.dot(xp)

        delta = r + self.gamma * vp - v
        delta_hat = self.ym.dot(x)

        dw = rho * (delta_hat * x - self.gamma * delta_hat * xp)
        dh = (rho * delta - delta_hat) * x

        self.theta = self.thetam + self.alpha * dw
        self.y= self.ym + self.eta * self.alpha * dh

        self.av_theta = (self.alpha * self.theta + self.alpha_1_tm1 * self.av_theta ) / (self.alpha+self.alpha_1_tm1)
        self.alpha_1_tm1 += self.alpha

    def getWeights(self):
        return self.av_theta

    def initWeights(self, u):
        self.theta = u
        self.av_theta = u
