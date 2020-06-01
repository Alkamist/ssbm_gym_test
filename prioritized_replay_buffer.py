import random
import numpy as np
from sumtree import SumTree
from collections import namedtuple, deque


Transition = namedtuple("Transition", field_names=["state", "action", "next_state", "reward"])


class PrioritizedReplayBuffer:
    def __init__(self, capacity, batch_size, epsilon=0.01, alpha=0.6, beta=1.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size
        #self.beta_increment_per_sampling = 0.001
        self.length = 0

    def __len__(self):
        return self.length

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def get_average_priority(self):
        if self.length > 0:
            return self.tree.total() / self.length
        else:
            return self.epsilon

    def add_by_priority(self, priority, state, action, next_state, reward):
        self.tree.add(priority, Transition(state, action, next_state, reward))
        self.length = min(self.length + 1, self.capacity)

#    def add(self, error, state, action, next_state, reward):
#        p = self._get_priority(error)
#        self.tree.add(p, Transition(state, action, next_state, reward))
#        self.length = min(self.length + 1, self.capacity)

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        #self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        importance_sampling_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        return Transition(*zip(*batch)), idxs, importance_sampling_weights

    def update_priority(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
