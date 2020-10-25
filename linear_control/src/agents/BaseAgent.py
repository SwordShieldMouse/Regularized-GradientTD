import numpy as np
from PyExpUtils.utils.random import sample
from src.utils.arrays import argmax
from src.utils.ReplayBuffer import ReplayBuffer
from src.utils.PER import PrioritizedReplayMemory

class BaseAgent:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        self.buffer_type = params.get('buffer_type', 'standard')
        self.buffer_size = params.get('buffer_size', 1)
        self.replay_steps = params.get('replay_steps', 0)
        self.prioritization = params.get('prioritization', 2.0)
        self.warmup = params.get('warmup', self.replay_steps * 10)

        if self.buffer_type == 'standard':
            self.buffer = ReplayBuffer(self.buffer_size)
        else:
            self.buffer = PrioritizedReplayMemory(self.buffer_size, alpha=self.prioritization)

        self.steps = 0

        # most agents have 2 sets of weights
        self.paramShape = (2,actions,features)

    def batch_update(self, gen, num):
        exps = gen.sample(samples=num)
        for i in range(num):
            x, a, xp, r, gamma = exps[i]
            self.update(x, a, xp, r, gamma)

    def policy(self, x, w=None):
        if w is None:
            w = self.getWeights()

        max_acts = argmax(w.dot(x))
        pi = np.zeros(self.actions)
        uniform = self.epsilon / self.actions
        pi += uniform

        # if there are no max acts, then we've diverged and everything is NaNs
        if len(max_acts) > 0:
            pi[max_acts] += (1.0 / len(max_acts)) - self.epsilon

        return pi

    def selectAction(self, x):
        return sample(self.policy(x, self.getWeights()))

    def applyUpdate(self, x, a, xp, r, gamma):
        return None, None

    # def update(self, x, a, xp, r, gamma):
    #     self.steps += 1
    #     self.buffer.add((x, a, xp, r, gamma))

    #     if self.replay_steps == 0:
    #         self.applyUpdate(x, a, xp, r, gamma)

    #     if self.steps < self.warmup:
    #         return

    #     for _ in range(self.replay_steps):
    #         sample, idxes = self.buffer.sample(1)
    #         _, delta = self.applyUpdate(*sample[0])
    #          self.buffer.update_priorities(idxes, [delta])

    def update(self, x, a, xp, r, gamma):
        self.steps += 1
        self.buffer.add((x, a, xp, r, gamma))

        self.applyUpdate(x, a, xp, r, gamma)

    def replay(self):
        if self.steps < self.warmup:
            return

        #self.steps = 0

        for _ in range(self.replay_steps):
            sample, idxes = self.buffer.sample(1)
            _, delta = self.applyUpdate(*sample[0])
            self.buffer.update_priorities(idxes, [delta])

    def getWeights(self):
        return self.w
