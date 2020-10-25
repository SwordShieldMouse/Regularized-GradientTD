import numpy as np

from src.problems.BaseProblem import BaseProblem
from src.environments.MountainCar import MountainCar as MCEnv
from src.environments.MountainCar import BACK, STAY, FORWARD
from PyFixedReps.TileCoder import TileCoder
from src.utils.policies import Policy
from PyFixedReps.BaseRepresentation import BaseRepresentation

class RBF(BaseRepresentation):
    def __init__(self, params):
        self.centers = np.array(params['centers'])
        #self.sigma = np.array(params['sigma'])
        self.sigma = 1.0/np.array(params['sigma'])

    def features(self):
        return len(self.centers)

    def encode(self, s, a = None):
        features = np.zeros(self.features())
        diff = s - self.centers
        squared = np.sum(diff * diff * self.sigma, axis=1)
        features = np.exp(-1 * squared /2)

        return features

class ScaledTileCoder(TileCoder):
    def encode(self, s):
        p = s[0]
        v = s[1]

        p = (p + 1.2) / 1.7
        v = (v + 0.07) / 0.14
        return super().encode((p, v)) / float(self.num_tiling)

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MCEnv()
        self.actions = 3

        self.rep = self.buildRep()

        self.features = self.rep.features()
        self.gamma = 0.99
        self.max_steps = exp.max_steps

class MountainCarTC(MountainCar):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)

    def buildRep(self):
        return  ScaledTileCoder({
            'dims': 2,
            'tiles': 4,
            'tilings': 16,
        })

class MountainCarRBF(MountainCar):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)

    def buildRep(self):
        N1=8
        N2=8
        ps = np.linspace(-1.2, 0.5, N1)
        vs = np.linspace(-0.07,0.07,N2)
        centers = np.array([[p,v] for p in ps for v in vs])
        return RBF({'sigma': np.array([2*1.7/(N1-1), 2*0.14/(N2-1)]), 'centers':centers})

def bangbang(s):
    if np.random.rand() < 0.1:
        return np.ones(3)/3
    a = np.zeros(3)
    if s[1]<0:
        a[BACK] = 1.0
    else:
        a[FORWARD] = 1.0
    return a

class OfflineMountainCar(MountainCar):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.evalSteps = exp.evalSteps
        self.epochs = exp.epochs

        self.behavior = Policy(bangbang)
