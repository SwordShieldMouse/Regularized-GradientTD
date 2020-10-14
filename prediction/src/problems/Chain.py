import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.environments.Chain import Chain
from src.utils.rlglue import OffPolicyWrapper

from src.utils.policies import Policy

class BaseChain(BaseProblem):
    def _getTarget(self):
        return Policy(lambda s: [])

    def _getRepresentation(self, n):
        return BaseRepresentation()

    def _getSize(self):
        return 19

    def _getdb(self):
        return np.array([
            0.0099983, 0.019997, 0.0299957, 0.0399956, 0.0499955, 0.0599974,
            0.0699993, 0.080004, 0.0900087, 0.100017, 0.0900087, 0.080004,
            0.0699993, 0.0599974, 0.0499955, 0.0399956, 0.0299957, 0.019997,
            0.0099983, 0
        ])

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        N = self._getSize()

        self.reward_scale = self.metaParameters.get('reward_scale', 1)
        self.env = Chain(N, self.reward_scale)

        # build target policy
        self.target = self._getTarget()

        self.behavior = Policy(lambda s: [0.5, 0.5])

        self.v_star = self.compute_v(N, self.target, self.reward_scale)

        # build representation
        self.rep = self._getRepresentation(N)

        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # compute the observable value for each state once
        self.X = np.array([
            self.rep.encode(i) for i in range(N + 1)
        ])

        # (1/n+1) sum_{k=0}^n P^k gives a matrix with db in each row, where P is the markov chain
        # induced by the behaviour policy
        self.db = self._getdb()

        # build transition probability matrix (under target)
        self.P = np.zeros((N + 1, N + 1))
        pl, pr = self.target.probs(0)
        self.P[0, 1] = pr
        self.P[0, N] = pl
        self.P[N-1, N-2] = pl
        self.P[N-1, N] = pr
        for i in range(1, N-1):
            self.P[i, i - 1] = pl
            self.P[i, i + 1] = pr

        self.R = np.zeros(N + 1)
        self.R[0] = pl * -self.reward_scale
        self.R[N-1] = pr * self.reward_scale

        self.setupIdealH()

    def getTarget(self, n):
        return NotImplementedError()

    def getGamma(self):
        return 1.0

    def getSteps(self):
        return self.exp.steps

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def compute_v(self, nstates, targetPolicy, reward_scale):
        gamma = self.getGamma()
        theta = 1e-8

        V = np.zeros(nstates + 1)

        delta = np.infty
        i = 0
        while delta > theta:
            i += 1
            delta = 0.0
            for s in range(nstates):
                p_left, p_right = targetPolicy.probs(s)

                v = V[s]

                right = s + 1
                if right >= nstates:
                    right_reward = reward_scale
                    right_value = 0
                else:
                    right_reward = 0.0
                    right_value = V[right]

                left = s - 1
                if left<0:
                    left_reward = -reward_scale
                    left_value = 0.0
                else:
                    left_reward = 0.0
                    left_value = V[left]

                V[s] = p_right * (right_reward + gamma * right_value) +\
                       p_left * (left_reward + gamma * left_value)

                delta = max(delta, np.abs(v - V[s]))

        return V

# ----------------
# -- Off-policy --
# ----------------

class Policy5050:
    def _getTarget(self):
        return Policy(lambda s: [.5, .5])

class Policy4060:
    def _getTarget(self):
        return Policy(lambda s: [.4, .6])

class Policy2575:
    def _getTarget(self):
        return Policy(lambda s: [.25, .75])

class Policy1090:
    def _getTarget(self):
        return Policy(lambda s: [.1, .9])

# --------------------
# -- Representation --
# --------------------

class RepInverted:
    def _getRepresentation(self, n):
        return Inverted(n)

class RepTabular:
    def _getRepresentation(self, n):
        return Tabular(n)

class RepDependent:
    def _getRepresentation(self, n):
        return Dependent(n)

# ---------------
# -- Resultant --
# ---------------

class ChainInverted5050(Policy5050, RepInverted, BaseChain):
    pass

class ChainInverted4060(Policy4060, RepInverted, BaseChain):
    pass

class ChainInverted2575(Policy2575, RepInverted, BaseChain):
    pass

class ChainTabular5050(Policy5050, RepTabular, BaseChain):
    pass

class ChainTabular4060(Policy4060, RepTabular, BaseChain):
    pass

class ChainTabular2575(Policy2575, RepTabular, BaseChain):
    pass

class ChainDependent5050(Policy5050, RepDependent, BaseChain):
    pass

class ChainDependent4060(Policy4060, RepDependent, BaseChain):
    pass

class ChainDependent2575(Policy2575, RepDependent, BaseChain):
    pass


# --------------------
# -- Representation --
# --------------------

class Inverted(BaseRepresentation):
    def __init__(self, N):
        m = np.ones((N,N)) - np.eye(N)

        self.map = np.zeros((N+1, N))
        self.map[:N] = (m.T / np.linalg.norm(m, axis = 1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Tabular(BaseRepresentation):
    def __init__(self, N):
        m = np.eye(N)

        self.map = np.zeros((N+1, N))
        self.map[:N] = m

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Dependent(BaseRepresentation):
    def __init__(self, N):
        nfeats = int(np.floor(N/2) + 1)
        self.map = np.zeros((N+1,nfeats))

        idx = 0
        for i in range(nfeats):
            self.map[idx,0:i+1] = 1
            idx+=1

        for i in range(nfeats-1,0,-1):
            self.map[idx,-i:] = 1
            idx+=1

        self.map[:N] = (self.map[:N].T / np.linalg.norm(self.map[:N], axis = 1)).T

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]
