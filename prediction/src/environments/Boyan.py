import numpy as np
from RlGlue import BaseEnvironment

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
