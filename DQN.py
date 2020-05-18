import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        #self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#class ResidualBlock(nn.Module):
#    def __init__(self, input_size, hidden_size):
#        super(ResidualBlock, self).__init__()
#        self.mlp = nn.Sequential(
#            nn.Linear(input_size, hidden_size),
#            nn.ReLU(True),
#            nn.Linear(hidden_size, input_size),
#            nn.ReLU(True)
#        )
#
#    def forward(self, x):
#        return x + self.mlp(x)
#
#class ResNet(nn.Module):
#    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2):
#        super(ResNet, self).__init__()
#        residual_blocks = [ResidualBlock(input_size, hidden_size) for _ in range(num_layers)]
#        self.residual_blocks = nn.Sequential(*residual_blocks)
#        self.linear_layer = nn.Linear(input_size, output_size)
#
#    def forward(self, x):
#        features = self.residual_blocks(x)
#        prediction = self.linear_layer(features)
#        return prediction

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
    def __init__(self, state_size, action_size, lr=0.00001, batch_size=32, memory_size=1000000,
                 update_every=40000, gamma=0.99, tau=0.003, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=100000, learn_every=16):
        self.state_size = state_size
        self.action_size = action_size
        self.update_every = update_every
        self.learn_every = learn_every
        self.batch_size = batch_size
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
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size, batch_size)
        self.current_step = 0
        self.loss_criterion = torch.nn.SmoothL1Loss()

    def step(self, state, action, reward, next_state, done):
        if self.current_step % self.learn_every == 0:
            if len(self.memory) > self.batch_size:
                self._learn()
        self.memory.add(state, action, reward, next_state, done)
        self.current_step += 1

    def act(self, state):
        output = self._get_output(state)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.0 * self.current_step / self.epsilon_decay)
        return output

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def evaluate(self):
        self.policy_net.eval()

    def _get_output(self, state):
        if random.random() > self.epsilon:
            state = torch.as_tensor([state], device=device).float()
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            return torch.argmax(action_values).item()
        else:
            return random.randrange(self.action_size)

    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

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

        if self.current_step % self.update_every == 0:
            self._hard_update()
        #self._soft_update()

    def _hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)