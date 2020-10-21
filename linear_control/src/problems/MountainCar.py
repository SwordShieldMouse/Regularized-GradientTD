import numpy as np

from src.problems.BaseProblem import BaseProblem
from src.environments.MountainCar import MountainCar as MCEnv
from src.environments.MountainCar import BACK, STAY, FORWARD
from PyFixedReps.TileCoder import TileCoder
from src.utils.policies import Policy

class ScaledTileCoder(TileCoder):
    def __init__(self, params):
        super().__init__(params)

    def encode(self, s, a):
        p = s[0]
        v = s[1]

        p = (p + 1.2) / 1.7
        v = (v + 0.07) / 0.14
        return super().encode((p, v), a) / float(self.num_tiling)

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MCEnv()
        self.actions = 3

        self.rep = TileCoder({
            'dims': 2,
            'tiles': 4,
            'tilings': 4,
            'actions': 3,
        })

        self.features = self.rep.features()
        self.gamma = 0.99
        self.max_steps = exp.max_steps

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
        self.evalEpisodes = exp.evalEpisodes
        self.epochs = exp.epochs

        self.behavior = Policy(bangbang)
        self.target = None
