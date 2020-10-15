import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation

def contains(m, vec):
    for r in m:
        if np.allclose(r, vec):
            return True

    return False

class BinaryEncoder(BaseRepresentation):
    def __init__(self, active, bits, states):
        self.bits = bits

        m = []
        s = 0
        while len(m) < states and s < bits * 1000:
            s = s + 1
            active_idxs = np.random.choice(range(bits), size = active, replace = False)
            rep = np.zeros(bits)
            rep[active_idxs] = 1

            if contains(m, rep):
                continue

            m.append(rep)

        m.append(np.zeros(bits))

        self.map = m

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.bits
