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


#class ReplayBuffer(object):
#    def __init__(self, max_size, gamma=0.997):
#        self.storage = []
#        self.max_size = max_size
#        self.next_index = 0
#        self.gamma = gamma
#        self.n_step_buffer = deque(maxlen=5)
#
#    def __len__(self) -> int:
#        return len(self.storage)
#
#    def add(self, state, action, reward, next_state, done):
#        transition = Transition(state, action, reward, next_state, done)
#
#        #self._add_transition(transition)
#
#        self.n_step_buffer.append(transition)
#        self._add_transition(self._calculate_n_step_return())
#
#    def _add_transition(self, transition):
#        if self.next_index >= len(self.storage):
#            self.storage.append(transition)
#        else:
#            self.storage[self.next_index] = transition
#
#        self.next_index = (self.next_index + 1) % self.max_size
#
#    def _encode_sample(self, indices):
#        states, actions, rewards, next_states, dones = [], [], [], [], []
#
#        for i in indices:
#            data = self.storage[i]
#            state, action, reward, next_state, done = data
#            states.append(state)
#            actions.append(action)
#            rewards.append(reward)
#            next_states.append(next_state)
#            dones.append(done)
#
#        return Transition(states, actions, rewards, next_states, dones)
#
#    def _calculate_n_step_return(self):
#        output = 0
#
#        for idx in range(len(self.n_step_buffer)):
#            output += self.n_step_buffer[idx].reward * (self.gamma ** idx)
#
#        return Transition(self.n_step_buffer[0].state, self.n_step_buffer[0].action, output, self.n_step_buffer[-1].next_state, self.n_step_buffer[-1].done)
#
#    def sample(self, batch_size):
#        indices = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
#        return self._encode_sample(indices)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size):
        super(PrioritizedReplayBuffer, self).__init__(max_size)
        self.alpha = 0.9
        #self.beta_start = 0.4
        #self.beta_frames = 500
        self.epsilon = 0.01
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.times_sampled = 0

    def add(self, state, action, reward, next_state, done):
        self.priorities[self.next_index] = self.priorities.max() if self.storage else 1.0
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state, done)

    def sample(self, batch_size):
        #beta = min(1.0, self.beta_start + self.times_sampled * (1.0 - self.beta_start) / self.beta_frames)
        beta = 1.0

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

        self.times_sampled += 1

        return self._encode_sample(indices), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.storage)
