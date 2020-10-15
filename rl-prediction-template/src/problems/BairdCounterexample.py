import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.Baird import Baird
from src.utils.SampleGenerator import SampleGenerator

from src.utils.policies import Policy

class BairdRep(BaseRepresentation):
    def __init__(self):
        self.map = self.buildFeatureMatrix()

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

    def buildFeatureMatrix(self):
        return np.array([
            [1, 2, 0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 2, 0],
            [2, 0, 0, 0, 0, 0, 0, 1],
        ])

class BairdCounterexample(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.behavior = Policy(lambda s: [6/7, 1/7])
        self.target = Policy(lambda s: [0.0, 1.0])

        # build representation
        self.rep = BairdRep()
        #
        # build environment
        self.env = Baird()

        # build agent
        self.agent = self.Agent(self.rep.features(), 2, self.params)

        # initialize agent with starting weight parameters
        self.agent.initWeights(np.array([1, 1, 1, 1, 1, 1, 1, 10], dtype='float64'))

    def getGamma(self):
        return 0.99

    def getSteps(self):
        return 5000

    def sampleExperiences(self):
        clone = BairdCounterexample(self.exp, self.idx)
        gen = SampleGenerator(clone)
        return gen
