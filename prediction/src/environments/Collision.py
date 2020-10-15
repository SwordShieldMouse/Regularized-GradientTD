import numpy as np
from RlGlue import BaseEnvironment

# Constants
RIGHT = 0
RETREAT = 1

class Collision(BaseEnvironment):
    def __init__(self):
        self.states = 8
        self.state = 0

    def start(self, s = None):
        self.state = np.random.randint(4) if s is None else s
        return self.state

    def step(self, a):
        if a == RIGHT:
            self.state = self.state + 1
            if self.state == 8:
                self.state = self.start()
                return (1.0, self.state, True)

            return (0.0, self.state, False)

        elif a == RETREAT:
            self.start()
            return (0.0, self.state, True)

        raise NotImplementedError('Unknown action taken')
