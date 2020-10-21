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

    def setRepresentation(self, rep):
        self.rep = rep

    def batch_update(self, gen, num):
        exps = gen.sample(samples=num)
        shape = (num,) + self.paramShape
        for i in range(num):
            grad = self.grads(*exps[i])
            self._apply(*grad, exps[i][1])

    def encode(self, s):
        print(s)
        return np.array([self.rep.encode(s,a) for a in range(self.actions)])

    def Q(self, s, a, w):
        return sum(w[self.rep.get_indices(s,a)])

    def Qs(self, s, w):
        return [self.Q(s,a,w) for a in range(self.actions)]
       
    def policy(self, s, w=None):
        if w is None:
            w = self.getWeights()

        max_acts = argmax(self.Qs(s,w))
        pi = np.zeros(self.actions)
        uniform = self.epsilon / self.actions
        pi += uniform

        # if there are no max acts, then we've diverged and everything is NaNs
        if len(max_acts) > 0:
            pi[max_acts] += (1.0 / len(max_acts)) - self.epsilon

        return pi

    def selectAction(self, s):
        return sample(self.policy(s, self.getWeights()))

    def applyUpdate(self, x, a, xp, r, gamma):
        return None, None

    def update(self, s, a, sp, r, gamma):
        self.steps += 1
        self.buffer.add((s, a, sp, r, gamma))

        if self.replay_steps == 0:
            self.applyUpdate(s, a, sp, r, gamma)

        if self.steps < self.warmup:
            return

        for _ in range(self.replay_steps):
            sample, idxes = self.buffer.sample(1)
            _, delta = self.applyUpdate(*sample[0])
            self.buffer.update_priorities(idxes, [delta])

    def getWeights(self):
        return self.w
