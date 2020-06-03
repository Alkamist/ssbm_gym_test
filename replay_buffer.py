import random
from collections import namedtuple

import numpy as np

from segment_tree import SumSegmentTree, MinSegmentTree


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
    def __init__(self, max_size, alpha=0.9, beta=1.0):
        super(PrioritizedReplayBuffer, self).__init__(max_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.01
        self.max_priority = 1.0

        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2

        self._tree_sum = SumSegmentTree(tree_capacity)
        self._tree_min = MinSegmentTree(tree_capacity)

    def add(self, state, action, reward, next_state, done):
        idx = self.next_index
        super().add(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self._tree_sum[idx] = priority
        self._tree_min[idx] = priority

    def sample(self, batch_size):
        mass = []
        weights = []
        total = self._tree_sum.sum(0, len(self.storage) - 1)
        mass = np.random.random(size=batch_size) * total
        indices = self._tree_sum.find_prefixsum_idx(mass)
        p_min = self._tree_min.min() / self._tree_sum.sum()
        max_weight = (p_min * len(self.storage)) ** (-self.beta)
        p_sample = self._tree_sum[indices] / self._tree_sum.sum()
        weights = (p_sample * len(self.storage)) ** (-self.beta) / max_weight
        encoded_sample = self._encode_sample(indices)
        return encoded_sample, indices, weights

    def update_priorities(self, indices, absolute_errors):
        priorities = (absolute_errors + self.epsilon) ** self.alpha
        self._tree_sum[indices] = priorities
        self._tree_min[indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))