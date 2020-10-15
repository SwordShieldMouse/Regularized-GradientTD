import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

class NoiseMask(BaseRepresentation):
    def __init__(self, rep):
        self.rep = rep
        self.base_features = self.rep.features()

    def encode(self, s):
        r = self.rep.encode(s)
        n = np.random.normal(0, 0.1, size=self.base_features)
        return r + n

    def features(self):
        return self.base_features
