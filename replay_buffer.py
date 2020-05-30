import random
import pickle
from collections import namedtuple, deque

import torch

Transition = namedtuple("Transition", field_names=["state", "action", "next_state", "reward"])

class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add_rollout(self, rollout):
        for i in range(len(rollout)):
            state = torch.tensor([rollout.states[i]], dtype=torch.float32, device=self.device)
            action = torch.tensor([[rollout.actions[i]]], dtype=torch.long, device=self.device)
            next_state = torch.tensor([rollout.next_states[i]], dtype=torch.float32, device=self.device)
            reward = torch.tensor([[rollout.rewards[i]]], dtype=torch.float32, device=self.device)
            self.add(state, action, next_state, reward)

    def add(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, k=batch_size)
        return Transition(*zip(*transitions))

    def save(self, file_path):
        pickle.dump(self.memory, open(file_path, "wb"))

    def load(self, file_path):
        self.memory = pickle.load(open(file_path, "rb"))

    def __len__(self):
        return len(self.memory)