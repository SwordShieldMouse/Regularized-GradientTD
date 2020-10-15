import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

class NoiseFeature(BaseRepresentation):
    def __init__(self, rep):
        self.rep = rep

    def encode(self, s):
        r = self.rep.encode(s)
        n = np.random.normal()
        return np.append(r, n)

    def features(self):
        return self.rep.features() + 1
