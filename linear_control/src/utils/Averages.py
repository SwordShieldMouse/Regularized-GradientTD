import numpy as np

class WeightedAverage:
    def __init__(self, x0):
        self.t = 0.0
        self.S = 0.0
        self.x = np.zeros_like(x0)
        self.update(x0)

    def update(self, x):
        self._update()
        wt = self._get()
        self.x += wt * (x -self.x)

    def reset(self, x):
        self.x = np.zeros_like(x)
        self.t = 0.0
        self.S = 0.0
        self.update(x)

    def get(self):
        return self.x

    def _update(self):
        raise(NotImplementedError("WeightedAverage: update not implemented"))

    def _get(self):
        raise(NotImplementedError('WeightedAverage: get not implemented'))

class Uniform(WeightedAverage):
    def __init__(self, x0):
        super().__init__(x0)

    def _update(self):
        self.t+=1

    def _get(self):
        return 1.0 / self.t

class Sqrt(WeightedAverage):
    def __init__(self, x0):
        super().__init__(x0)

    def _update(self):
        self.t += 1
        self.S += np.sqrt(self.t)

    def _get(self):
        return np.sqrt(self.t) / self.S

class Linear(WeightedAverage):
    def __init__(self, x0):
        super().__init__(x0)

    def _update(self):
        self.t += 1
        self.S += self.t

    def _get(self):
        return  self.t / self.S
