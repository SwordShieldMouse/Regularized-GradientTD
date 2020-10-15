import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

class SparseRandomLinear(BaseRepresentation):
    def __init__(self, states, features, sparsity):
        self.feats = features

        W = np.random.normal(0, 0.1, (states, features))
        M = np.random.uniform(0, 1, (states, features))
        M[M < sparsity] = 0
        M[M >= sparsity] = 1

        W = W * M

        S = np.eye(states)

        self.map = S.dot(W)

        # make sure norm of features is consistent with other experiments
        state_norms = np.linalg.norm(self.map, axis=1)
        norm = np.tile(state_norms, [features, 1]).T
        self.map = self.map / norm

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.feats
