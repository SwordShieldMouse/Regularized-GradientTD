import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.utils.Critterbot import loadTiles

from src.environments.Critterbot import Critterbot as CritterbotEnv


class CritterbotDataRep(BaseRepresentation):
    def __init__(self):
        self.tiles = loadTiles()
        self._features = 8197
        self.numActive = float(457)

    def encode(self, s):
        f=np.zeros(self._features)
        f[self.tiles[s]] = 1.0
        return f

    def features(self):
        return self._features


class Critterbot(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.sensorIdx = self.params["sensorIdx"]
        self.env = CritterbotEnv(self.sensorIdx)
        self.rep = CritterbotDataRep()

        # build agent
        self.agent = self.Agent(self.rep.features(), 0, self.params)
        self.steps = exp.steps
        self.gamma = self.params['gamma']

    def getGamma(self):
        return self.gamma

    def getSteps(self):
        return self.steps
