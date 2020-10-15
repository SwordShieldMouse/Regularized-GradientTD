import numpy as np
from PyExpUtils.utils.fp import once
from src.problems.BaseProblem import BaseProblem
from src.environments.Collision import Collision as CollisionEnv
from src.representations.BinaryEncoder import BinaryEncoder
from src.utils.rl_glue import RlGlueCompatWrapper
from src.utils.SampleGenerator import SampleGenerator
import src.utils.policies as Policy

class Collision(BaseProblem):
    def _buildRepresentation(self, run):
        return BinaryEncoder(active=3, bits=6, states=8)

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        run = exp.getRun(idx)
        # build state distribution
        self.db = np.array([0.05715078, 0.1142799,  0.17142456, 0.22856842, 0.22856842, 0.11428067,
                            0.05715311, 0.02857415, 0.])
        # build true value function
        self.v_star = np.array([ np.power(self.getGamma(), 7 - i) for i in range(8) ] + [0])

        # build representation
        self.rep = self._buildRepresentation(run)
        # build environment
        self.env = CollisionEnv()
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build behavior policy
        self.behavior = Policy.fromStateArray(
            [[1.0, 0.0]] * 4 +
            [[0.5, 0.5]] * 4,
        )
        # build target policy
        self.target = Policy.fromActionArray([1.0, 0.0])

        # compute the observable value for each state once
        self.X = np.array([
            self.rep.encode(i) for i in range(9)
        ])

        self.P = np.zeros((9, 9))
        for i in range(8):
            self.P[i, i + 1] = 1.0

        self.R = np.zeros(9)
        self.R[7] = 1.0

        self.setupIdealH()

    def getGamma(self):
        return 0.9

    def getSteps(self):
        return 20000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        # wrap the agent so that its API is consistent with the RLGlue API
        # this is just a compatibility layer between disparate APIs
        return RlGlueCompatWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def sampleExperiences(self):
        clone = StandardCollision(self.exp, self.idx)
        clone.rep = self.rep
        gen = SampleGenerator(clone)
        return gen

    