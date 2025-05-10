import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, size=5000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, device="cpu", reward_bias=True):
        if not reward_bias:
            batch = random.sample(self.buffer, batch_size)
        else:
            # Compute sampling probabilities
            rewards = np.array([abs(tr[2]) for tr in self.buffer])
            probs = rewards + 1e-2  # Add small constant to avoid 0
            probs = probs / probs.sum()
            indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
            batch = [self.buffer[i] for i in indices]

        obs, act, rew, next_obs, done = zip(*batch)

        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(act), dtype=torch.int64, device=device),
            torch.tensor(np.array(rew), dtype=torch.float32, device=device),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=device),
            torch.tensor(np.array(done), dtype=torch.float32, device=device),
        )

