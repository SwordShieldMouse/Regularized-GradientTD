import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.RandomWalk import RandomWalk as RWEnv

from src.utils.policies import Policy

class RandomWalk(BaseProblem):
    def _buildRepresentation(self, name):
        if name == 'tabular':
            return Tabular(self.states)

        if name == 'inverted':
            return Inverted(self.states)

        if name == 'dependent':
            return Dependent(self.states)

        raise NotImplementedError('Unexpected representation name: ' + name)

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        self.states = self.params['states']

        pl = self.params['behavior']
        self.behavior = Policy(lambda s: [pl, 1.0-pl])

        pl = self.params['target']
        self.target = Policy(lambda s: [pl, 1.0-pl])

        self.env = RWEnv(self.states)

        # build representation
        representation = self.params['representation']
        self.rep = self._buildRepresentation(representation)

        # build agent
        self.agent = self.Agent(self.rep.features(), 2, self.params)

    def getGamma(self):
        return 1.0

# --------------------
# -- Representation --
# --------------------

class Inverted(BaseRepresentation):
    def __init__(self, N):
        self.N = N
        self.map = self.buildFeatureMatrix()

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

    def buildFeatureMatrix(self):
        N = self.N
        invI = np.ones((N, N)) - np.eye(N)
        m = np.zeros((N+1, N))
        m[:N] = (invI.T / np.linalg.norm(invI, axis=1)).T
        return m


class Tabular(BaseRepresentation):
    def __init__(self, N):
        self.N = N
        self.map = self.buildFeatureMatrix()

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

    def buildFeatureMatrix(self):
        N = self.N
        m = np.zeros((N+1, N))
        m[:N] = np.eye(N)
        return m

class Dependent(BaseRepresentation):
    def __init__(self, N):
        self.N = N
        self.map = self.buildFeatureMatrix()

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

    def buildFeatureMatrix(self):
        N = self.N
        nfeats = int(np.floor(N/2) + 1)
        m = np.zeros((N+1, nfeats))

        idx = 0
        for i in range(nfeats):
            m[idx, 0:i+1] = 1
            idx += 1

        for i in range(nfeats-1, 0, -1):
            m[idx, -i:] = 1
            idx += 1

        m[:N] = (m[:N].T / np.linalg.norm(m[:N], axis=1)).T
        return m
