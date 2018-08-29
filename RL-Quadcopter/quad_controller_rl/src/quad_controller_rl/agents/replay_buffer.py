"""Replay Buffer."""

import random
from collections import deque

import numpy as np

class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.size = size
        self.memory = deque(maxlen=self.size)
        self.len = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = (state, action, reward, next_state, int(done))

        self.len += 1
        if self.len > self.size:
            self.len = self.size

        self.memory.append(experience)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        batch_size = min(batch_size, self.__len__())
        batch = random.sample(self.memory, batch_size)

        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch]
        dones = [e[4] for e in batch]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return self.len
