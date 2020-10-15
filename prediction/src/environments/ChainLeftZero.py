import numpy as np
from RlGlue import BaseEnvironment

# Constants
LEFT = 0
RIGHT = 1

class ChainLeftZero(BaseEnvironment):
    def __init__(self, size=19):
        self.states = size
        self.state = size // 2

    def start(self):
        self.state = self.states // 2
        return self.state

    def step(self, a):
        if a == LEFT:
            self.state = max(self.state - 1, -1)

        elif a == RIGHT:
            self.state = min(self.state + 1, self.states)

        reward = 0
        terminal = False

        if self.state == -1:
            terminal = True

        elif self.state == self.states:
            reward = 1
            terminal = True

        return (reward, self.state, terminal)
