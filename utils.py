import random
import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = collections.deque(maxlen=max_size)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)