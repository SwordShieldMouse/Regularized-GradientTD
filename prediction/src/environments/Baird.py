import numpy as np
from RlGlue import BaseEnvironment

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
