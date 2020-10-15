import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.Boyan import Boyan as BoyanEnv
from src.utils.SampleGenerator import SampleGenerator

from src.utils.policies import fromStateArray

class BoyanRep(BaseRepresentation):
    def __init__(self):
        self.map = self.buildFeatureMatrix()

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

    def buildFeatureMatrix(self):
        return np.array([
            [1,    0,    0,    0   ],
            [0.75, 0.25, 0,    0   ],
            [0.5,  0.5,  0,    0   ],
            [0.25, 0.75, 0,    0   ],
            [0,    1,    0,    0   ],
            [0,    0.75, 0.25, 0   ],
            [0,    0.5,  0.5,  0   ],
            [0,    0.25, 0.75, 0   ],
            [0,    0,    1,    0   ],
            [0,    0,    0.75, 0.25],
            [0,    0,    0.5,  0.5 ],
            [0,    0,    0.25, 0.75],
            [0,    0,    0,    1   ],
            [0,    0,    0,    0   ],
        ])

class Boyan(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        # build target policy
        self.target = fromStateArray(
            # [P(RIGHT), P(SKIP)]
            [[.5, .5]] * 11 +
            [[1, 0]] * 2
        )

        # on-policy version of this domain
        self.behavior = self.target

        # build representation
        self.rep = BoyanRep()
        # build environment
        self.env = BoyanEnv()
        # build agent
        self.agent = self.Agent(self.rep.features(), 2, self.params)

    def getGamma(self):
        return 1.0

    def getSteps(self):
        return 10000

    def sampleExperiences(self):
        clone = Boyan(self.exp, self.idx)
        gen = SampleGenerator(clone)
        return gen
