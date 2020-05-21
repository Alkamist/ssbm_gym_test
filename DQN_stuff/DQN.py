import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

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

class DQN():
    def __init__(self, state_size, action_size, lr=0.00001, batch_size=32, gamma=0.99, target_update_frequency=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.loss_criterion = torch.nn.SmoothL1Loss()
        self.target_update_frequency = target_update_frequency
        self.learn_iterations = 0

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                action = self.policy_net(state).max(1)[1].view(1, 1)
            self.policy_net.train()
            return action
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def save(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.target_net.load_state_dict(torch.load(file_path))

    def evaluate(self):
        self.policy_net.eval()

    def learn(self, batch, batch_size):
        self.policy_net.train()
        self.target_net.eval()

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward, 1)

        non_final_mask = torch.as_tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch[0]

        loss = self.loss_criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network.
        if self.learn_iterations % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_iterations += 1

    #def _soft_update(self, tau):
    #    for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
    #        target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)