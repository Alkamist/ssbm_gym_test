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
            state = rollout.states[i]
            action = rollout.actions[i]
            next_state = rollout.next_states[i]
            reward = rollout.rewards[i]
            self.add(state, action, next_state, reward)

    def add(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, k=batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        return Transition(state_batch, action_batch, next_state_batch, reward_batch)

    def save(self, file_path):
        pickle.dump(self.memory, open(file_path, "wb"))

    def load(self, file_path):
        self.memory = pickle.load(open(file_path, "rb"))

    def __len__(self):
        return len(self.memory)