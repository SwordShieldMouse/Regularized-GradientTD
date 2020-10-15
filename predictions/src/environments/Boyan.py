import numpy as np
from RlGlue import BaseEnvironment
from src.utils.errors import getSteadyStateDist

# Constants
RIGHT = 0
SKIP = 1

class Boyan(BaseEnvironment):
    def __init__(self):
        self.states = 13
        self.state = 0

    def start(self):
        self.state = 0
        return self.state

    def step(self, a):
        reward = -3
        terminal = False

        if a == SKIP and self.state > 10:
            print("Double right action is not available in state 11 or state 12... Exiting now.")
            exit()

        if a == RIGHT:
            self.state = self.state + 1
        elif a == SKIP:
            self.state = self.state + 2

        if (self.state == 13):
            terminal = True
            reward = -2

        return (reward, self.state, terminal)

    def buildTransitionMatrix(self, policy):
        P = np.zeros((14, 14))
        for i in range(11):
            P[i, i+1] = .5
            P[i, i+2] = .5

        P[11, 12] = 1
        P[12, 13] = 1
        return P

    def buildAverageReward(self, policy):
        return np.array([-3] * 12 + [-2, 0])

    def getSteadyStateDist(self, policy):
        return np.array([0.07757417, 0.07680082, 0.0768048, 0.07680444, 0.0767995, 0.07680488, 0.07680497, 0.07680541, 0.0768076, 0.07680623, 0.07680757, 0.07680545, 0.07757417, 0])
