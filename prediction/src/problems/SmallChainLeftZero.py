import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem
from src.problems.Chain import Inverted, Dependent, Tabular
from src.environments.ChainLeftZero import ChainLeftZero as ChainEnv
from src.utils.rl_glue import RlGlueCompatWrapper
from src.utils.policies import Policy

class BaseChainLeftZero(BaseProblem):
    def _getTarget(self):
        return Policy(lambda s: [])

    def _getRepresentation(self, n):
        return BaseRepresentation()

    def _getSize(self):
        return 5

    def _getdb(self):
        return np.array([0.111111, 0.222222, 0.333333, 0.222222, 0.111111, 0])

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        N = self._getSize()

        self.env = ChainEnv(N)

        # build representation
        self.rep = self._getRepresentation(N)

        # build environment
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build target policy
        self.target = self._getTarget()

        self.behavior = Policy(lambda s: [0.5, 0.5])

        # compute the observable value for each state once
        self.X = np.array([
            self.rep.encode(i) for i in range(N + 1)
        ])

        # (1/n+1) sum_{k=0}^n P^k gives a matrix with db in each row, where P is the markov chain
        # induced by the behaviour policy
        self.db = self._getdb()

        self.v_star = self.compute_v(N, self.target)

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

        self.R = np.zeros(N+1)
        # self.R[0] = pl * -1
        self.R[N-1] = pr * 1

        self.setupIdealH()

    def getTarget(self, n):
        return NotImplementedError()

    def getGamma(self):
        return 1.0

    def getSteps(self):
        return 1000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return RlGlueCompatWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def compute_v(self, nstates, targetPolicy):
        gamma = self.getGamma()
        theta = 1e-8

        V = np.zeros(nstates+1)

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
                    right_reward = 1
                    right_value = 0
                else:
                    right_reward = 0.0
                    right_value = V[right]

                left = s - 1
                if left<0:
                    left_reward = 0
                    left_value = 0.0
                else:
                    left_reward = 0.0
                    left_value = V[left]

                V[s] = p_right * (right_reward + gamma * right_value) +\
                       p_left * (left_reward + gamma * left_value)

                delta = max(delta, np.abs(v - V[s]))

        return V

# ----------------
# -- On-policy --
# ----------------

class Chain5050LeftZero(BaseChainLeftZero):
    def _getTarget(self):
        return Policy(lambda s: [.5, .5])

# --------------------
# -- Representation --
# --------------------

class ChainInvertedLeftZero(BaseChainLeftZero):
    def _getRepresentation(self, n):
        return Inverted(n)

    def getSteps(self):
        return 500


class ChainTabularLeftZero(BaseChainLeftZero):
    def _getRepresentation(self, n):
        return Tabular(n)

    def getSteps(self):
        return 200

class ChainDependentLeftZero(BaseChainLeftZero):
    def _getRepresentation(self, n):
        return Dependent(n)

    def getSteps(self):
        return 400

# ---------------
# -- Resultant --
# ---------------

class SmallChainDependent5050LeftZero(Chain5050LeftZero, ChainDependentLeftZero):
    pass

class SmallChainInverted5050LeftZero(Chain5050LeftZero, ChainInvertedLeftZero):
    pass

class SmallChainTabular5050LeftZero(Chain5050LeftZero, ChainTabularLeftZero):
    pass
