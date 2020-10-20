import numpy as np

from src.problems.BaseProblem import BaseProblem
from src.environments.MountainCar import MountainCar as MCEnv
from src.environments.MountainCar import BACK, STAY, FORWARD
from PyFixedReps.TileCoder import TileCoder
from src.utils.policies import Policy

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

        self.rep = ScaledTileCoder({
            'dims': 2,
            'tiles': 4,
            'tilings': 16,
        })

        self.features = self.rep.features()
        self.gamma = 0.99
        self.max_steps = exp.max_steps


class OfflineMountainCar(MountainCar):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.evalSteps = exp.evalSteps
        self.evalEpisodes = exp.evalEpisodes

        def pi(s):
            a = np.zeros(self.actions)
            if s[1]<0:
                a[BACK] = 1.0
            else:
                a[FORWARD] = 1.0
            return a

        self.behavior = Policy(pi)
        self.target = Policy(lambda s: self.getAgent().policy(self.rep.encode(s)))
