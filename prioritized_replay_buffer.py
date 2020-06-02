import random
import numpy as np
from collections import namedtuple


Transition = namedtuple("Transition", field_names=["state", "action", "next_state", "reward"])


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.alpha = 0.6
        self.beta = 1.0
        self.batch_size = batch_size
        self.capacity = capacity
        self.buffer = []
        self.write_position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, next_state, reward):
        max_priority = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.write_position] = max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, next_state, reward))
        else:
            self.buffer[self.write_position] = Transition(state, action, next_state, reward)

        self.write_position = (self.write_position + 1) % self.capacity

    def sample(self):
        buffer_length = len(self.buffer)

        if buffer_length == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.write_position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(buffer_length, self.batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = buffer_length
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return Transition(*zip(*samples)), indices, weights

    def update_priority(self, idx, error):
        self.priorities[idx] = (error + 0.01) ** self.alpha

    def __len__(self):
        return len(self.buffer)
