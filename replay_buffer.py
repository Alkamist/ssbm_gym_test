import random
from collections import deque

import numpy as np
import torch

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, max_size, n_step_size, gamma, device):
        self.storage = []
        self.max_size = max_size
        self.next_index = 0

        self.device = device

        self.gamma = gamma
        self.n_step_size = n_step_size
        self.n_step_states = deque(maxlen=self.n_step_size)
        self.n_step_actions = deque(maxlen=self.n_step_size)
        self.n_step_rewards = deque(maxlen=self.n_step_size)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state, action, reward, next_state, done):
        if self.n_step_size > 1:
            self.n_step_states.append(state)
            self.n_step_actions.append(action)
            self.n_step_rewards.append(reward)

            if len(self.n_step_states) >= self.n_step_size:
                n_step_state = self.n_step_states.popleft()
                n_step_action = self.n_step_actions.popleft()
                n_step_reward = self._get_n_step_return()
                self.n_step_rewards.popleft()
            else:
                n_step_state = state
                n_step_action = action
                n_step_reward = reward

            self._add(n_step_state, n_step_action, n_step_reward, next_state, done)

        else:
            self._add(state, action, reward, next_state, done)

    def _sample_from_indices(self, indices):
        batch_of_items = []
        for i in indices:
            batch_of_items.append(self.storage[i])
        return batch_of_items

    def sample_batch(self, batch_size):
        indices = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._sample_from_indices(indices)

    def _add(self, state, action, reward, next_state, done):
        item = (torch.tensor([state], dtype=torch.float32, device=self.device),
                torch.tensor([[action]], dtype=torch.long, device=self.device),
                torch.tensor([[reward]], dtype=torch.float32, device=self.device),
                torch.tensor([next_state], dtype=torch.float32, device=self.device),
                torch.tensor([[done]], dtype=torch.float32, device=self.device))

        if self.next_index >= len(self.storage):
            self.storage.append(item)
        else:
            self.storage[self.next_index] = item

        self.next_index = (self.next_index + 1) % self.max_size

    def _get_n_step_return(self):
        return np.sum([r * (self.gamma ** i) for i, r in enumerate(self.n_step_rewards)])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, n_step_size, gamma, device):
        super(PrioritizedReplayBuffer, self).__init__(max_size, n_step_size, gamma, device)
        self.alpha = 0.9
        self.beta = 0.4
        self.epsilon = 0.01

        it_capacity = 1
        while it_capacity < max_size:
            it_capacity *= 2

        self.it_sum = SumSegmentTree(it_capacity)
        self.it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        priority = self.max_priority ** self.alpha
        self.it_sum[self.next_index] = priority
        self.it_min[self.next_index] = priority
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state, done)

    def sample_batch(self, batch_size):
        assert self.beta > 0

        storage_length = len(self.storage)
        p_total = self.it_sum.sum(0, storage_length - 1)
        every_range_len = p_total / batch_size

        indices = []
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.it_sum.find_prefixsum_idx(mass)
            indices.append(idx)

        p_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (p_min * storage_length) ** (-self.beta)

        weights = []
        for idx in indices:
            p_sample = self.it_sum[idx] / self.it_sum.sum()
            weight = (p_sample * storage_length) ** (-self.beta)
            weights.append(weight / max_weight)

        return self._sample_from_indices(indices), torch.tensor(weights, dtype=torch.float32, device=self.device), indices

    def update_priorities_from_errors(self, indices, errors):
        for idx, error in zip(indices, errors):
            assert error > 0
            assert 0 <= idx < len(self.storage)
            priority = error ** self.alpha
            self.it_sum[idx] = priority
            self.it_min[idx] = priority
            self.max_priority = max(self.max_priority, priority)
