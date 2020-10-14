import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.Boyan import Boyan as BoyanEnv
from src.utils.rlglue import OffPolicyWrapper
from src.utils.SampleGenerator import SampleGenerator

import src.utils.policies as Policy

class BoyanRep(BaseRepresentation):
    def __init__(self):
        self.map = np.array([
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

    def encode(self, s):
        return self.map[s]

    def features(self):
        return 4

class Boyan(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx
        # build state distribution
        self.db = np.array([0.07757417, 0.07680082, 0.0768048, 0.07680444, 0.0767995, 0.07680488, 0.07680497, 0.07680541, 0.0768076, 0.07680623, 0.07680757, 0.07680545, 0.07757417, 0])
        # build true value function
        self.v_star = np.array([-24., -22., -20., -18., -16., -14., -12., -10., -8., -6., -4., -2., 0., 0.])

        # build representation
        self.rep = BoyanRep()
        # build environment
        self.env = BoyanEnv()
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build target policy
        self.target = Policy.fromStateArray(
            # [P(RIGHT), P(SKIP)]
            [[.5, .5]] * 11 +
            [[1, 0]] * 2
        )

        # on-policy version of this domain
        self.behavior = self.target

        # compute the observable value for each state once
        self.X = np.array([
            self.rep.encode(i) for i in range(len(self.db))
        ])

        # build transition probability matrix (under target policy) for computing ideal H
        self.P = np.zeros((14, 14))
        for i in range(11):
            self.P[i, i+1] = .5
            self.P[i, i+2] = .5

        self.P[11, 12] = 1
        self.P[12, 13] = 1

        # build Reward structure for computing ideal H
        self.R = np.array([-3] * 12 + [-2, 0])

        # always do this since we need it for RMSPBE
        # computes A, b, C
        self.setupIdealH()

    def getGamma(self):
        return 1.0

    def getSteps(self):
        return 10000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def sampleExperiences(self):
        clone = Boyan(self.exp, self.idx)
        gen = SampleGenerator(clone)
        return gen
