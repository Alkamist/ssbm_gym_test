import random
import pickle
from collections import namedtuple, deque


Transition = namedtuple("Transition", field_names=["state", "action", "next_state", "reward"])


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self):
        transitions = random.sample(self.memory, k=self.batch_size)
        return Transition(*zip(*transitions))

    def save(self, file_path):
        pickle.dump(self.memory, open(file_path, "wb"))

    def load(self, file_path):
        self.memory = pickle.load(open(file_path, "rb"))

    def __len__(self):
        return len(self.memory)