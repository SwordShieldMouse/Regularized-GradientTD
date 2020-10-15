import numpy as np
from RlGlue import BaseEnvironment
from src.utils.errors import getSteadyStateDist

# Constants
DASH = 0
SOLID = 1

class Baird(BaseEnvironment):
    def __init__(self):
        self.states = 7
        self.state = 0

    def start(self):
        self.state = 6
        return self.state

    def step(self, a):
        if a == SOLID:
            self.state = 6
        elif a == DASH:
            self.state = np.random.randint(6)

        return (0, self.state, False)

    # NOTE: problem assumes target policy = always solid
    def buildTransitionMatrix(self, policy):
        P = np.zeros((7, 7))
        P[:, 6] = 1
        return P

    def buildAverageReward(self, policy):
        return np.zeros(7)

    def getSteadyStateDist(self, policy):
        return np.ones(7) * (1/7)
