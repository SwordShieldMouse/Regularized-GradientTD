from environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

from problems.BaseProblem import BaseProblem
from PyFixedReps.TileCoder import TileCoder

class ScaledTileCoder(TileCoder):
    def __init__(self, params):
        super().__init__(params)
        self.mins =  np.array([-3, -3.5, -0.25, -3.5])
        self.maxes = np.array([3, 3.5, 0.25, 3.5])
        self.spans = self.maxes - self.mins

    def encode(self, s, a=None):
        sn = np.divide(s - self.mins, self.spans)
        return super().encode(sn) / float(self.num_tiling)

class CartPole(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 2

        self.features = 4
        self.gamma = 0.99

        self.rep = ScaledTileCoder({
            'dims': self.features,
            'tiles': 4,
            'tilings': 16,
        })

        self.max_episode_steps = 500

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Gym('CartPole-v1', self.run, self.max_episode_steps + 2)
