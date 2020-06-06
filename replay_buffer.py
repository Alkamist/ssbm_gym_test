import random
from collections import namedtuple

import numpy as np


Transition = namedtuple("Transition", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state, action, reward, next_state, done):
        data = Transition(state, action, reward, next_state, done)

        if self.next_index >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_index] = data

        self.next_index = (self.next_index + 1) % self.max_size

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            data = self.storage[i]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return Transition(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(indices)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size):
        super(PrioritizedReplayBuffer, self).__init__(max_size)
        self.alpha = 0.6
        #self.beta_start = 0.4
        #self.beta_frames = 500
        self.epsilon = 0.01
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        #self.times_sampled = 0

    def add(self, state, action, reward, next_state, done):
        self.priorities[self.next_index] = self.priorities.max() if self.storage else 1.0
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state, done)

    def sample(self, batch_size):
        #beta = min(1.0, self.beta_start + self.times_sampled * (1.0 - self.beta_start) / self.beta_frames)
        beta = 0.5

        storage_length = len(self.storage)

        if storage_length == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.next_index]

        probabilities = (priorities + self.epsilon) ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(storage_length, batch_size, p=probabilities)

        weights = (storage_length * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        #self.times_sampled += 1

        return self._encode_sample(indices), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.storage)
