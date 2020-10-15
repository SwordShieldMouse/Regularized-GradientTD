import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

from src.utils.arrays import last

def randomSparseMatrix(inputs, outputs, sparsity):
    W = np.random.normal(0, 0.1, (inputs, outputs))
    M = np.random.uniform(0, 1, (inputs, outputs))
    M[M < sparsity] = 0
    M[M >= sparsity] = 1

    return W * M

def relu(x):
    return np.where(x < 0, np.zeros_like(x), x)

class SparseNetwork(BaseRepresentation):
    def __init__(self, states, features, sparsity):
        self.feats = features

        maps = []
        inp = states
        for hidden in features:
            maps.append(randomSparseMatrix(inp, hidden, sparsity))
            inp = hidden

        S = np.eye(states)

        self.map = S
        for W in maps:
            self.map = relu(self.map.dot(W))

        state_norms = np.linalg.norm(self.map, axis=1)
        norm = np.tile(state_norms, [features, 1]).T
        self.map = self.map / norm

    def encode(self, s):
        return self.map[s]

    def features(self):
        return last(self.feats)
