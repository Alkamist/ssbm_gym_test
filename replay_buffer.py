import random
from collections import deque

import numpy as np
import torch


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

                self._add(n_step_state, n_step_action, n_step_reward, next_state, done)
        else:
            self._add(state, action, reward, next_state, done)

    def sample_batch(self, batch_size):
        indices = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

        batch_of_items = []
        for i in indices:
            batch_of_items.append(self.storage[i])

        return batch_of_items

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
