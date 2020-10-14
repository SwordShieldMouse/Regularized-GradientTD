import numpy as np

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.position = -1
        self._num_added = 0
        self.buffer = [None] * self.size

    def _nextPosition(self):
        self.position = (self.position + 1) % self.size
        return self.position

    def add(self, item):
        self._num_added += 1
        self.buffer[self._nextPosition()] = item

    # def replace(self, item):
    # 	self.buffer[self.position] = item

    def sample(self, samples = 1):
        # get a buffer with no None samples
        buffer = self.getAll()
        samples = samples if samples <= len(buffer) else len(buffer)
        indices = np.random.choice(range(len(buffer)), size=samples, replace=False)
        items = [buffer[i] for i in indices]
        return items

    def getAll(self):
        if self.isFull():
            return self.buffer

        return self.buffer[:self._num_added]

    def isFull(self):
        return self._num_added >= self.size

    def __len__(self):
        if self.isFull():
            return len(self.buffer)

        return self._num_added

class EndlessBuffer:
    def __init__(self, _ = None):
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)

    def sample(self, samples = 1):
        indices = np.random.choice(range(len(self.buffer)), size=samples, replace=False)
        items = [self.buffer[i] for i in indices]
        return items

    def getAll(self):
        return self.buffer

    def isFull(self):
        return False

    def __len__(self):
        return len(self.buffer)
