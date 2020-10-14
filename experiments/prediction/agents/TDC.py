import numpy as np
from agents.TDRC import TDRC
from utils.dict import merge

class TDC(TDRC):
    def __init__(self, features, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, merge(params, { 'beta': 0 }))

class BatchTDC(TDRC):
    def __init__(self, features, params):
        # TDC is just an instance of TDRC where beta = 0
        super().__init__(features, merge(params, { 'beta': 0 }))
        self.t = 0.0
        self.av_w = np.zeros(features)

    def update(self, x, a, r, xp, rho):
        self.t+=1.0
        super().update(x,a,r,xp,rho)
        self.av_w += 1.0 / self.t * (self.w - self.av_w)

    def initWeights(self, u):
        self.w = u
        self.av_w = u

    def getWeights(self):
        return self.av_w
