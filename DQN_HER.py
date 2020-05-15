import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.fc3(x)

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.as_tensor([e.state for e in experiences if e is not None], device=device).float()
        actions = torch.as_tensor([[e.action] for e in experiences if e is not None], device=device).long()
        rewards = torch.as_tensor([[e.reward] for e in experiences if e is not None], device=device).float()
        next_states = torch.as_tensor([e.next_state for e in experiences if e is not None], device=device).float()
        dones = torch.as_tensor([[e.done] for e in experiences if e is not None], device=device).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent():
    def __init__(self, state_size, action_size, lr=0.001, batch_size=16, memory_size=10000,
                 update_every=4, gamma=0.99, tau=0.003, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.996, HER_batch_size=16):
        self.state_size = state_size
        self.action_size = action_size
        self.update_every = update_every
        self.batch_size = batch_size
        self.HER_batch_size = HER_batch_size
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size, batch_size)
        self.HER_memory = ReplayBuffer(memory_size, HER_batch_size)
        self.current_step = 0
        #self.loss_criterion = torch.nn.MSELoss()
        self.loss_criterion = torch.nn.SmoothL1Loss()
        self.no_epsilon = False

    def step(self, state, action, reward, next_step, done):
        self.current_step = (self.current_step + 1) % self.update_every
        if self.current_step == 0:
            if (len(self.memory) > self.batch_size) and (len(self.HER_memory) > self.HER_batch_size):
                self._learn()
        self.memory.add(state, action, reward, next_step, done)
        if done:
            self.epsilon = self.epsilon_start

    def act(self, state):
        output = self._get_output(state)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        return output

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def evaluate(self):
        self.policy_net.eval()
        self.no_epsilon = True

    def _get_output(self, state):
        state = torch.as_tensor([state], device=device).float()

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        if self.no_epsilon or (random.random() > self.epsilon):
            return torch.argmax(action_values).item()
        else:
            return random.randrange(self.action_size)

    def _sample_memory(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        HER_states, HER_actions, HER_rewards, HER_next_states, HER_dones = self.HER_memory.sample()

        states = torch.cat((states, HER_states))
        actions = torch.cat((actions, HER_actions))
        rewards = torch.cat((rewards, HER_rewards))
        next_states = torch.cat((next_states, HER_next_states))
        dones = torch.cat((dones, HER_dones))

        return states, actions, rewards, next_states, dones

    def _learn(self):
        states, actions, rewards, next_states, dones = self._sample_memory()

        self.policy_net.train()
        self.target_net.eval()

        predicted_targets = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * labels_next * (1 - dones))

        loss = self.loss_criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #self._hard_update()
        self._soft_update()

    def _hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)