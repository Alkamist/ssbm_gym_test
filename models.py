import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Policy, self).__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.core_hidden_size = 64
        self.policy_hidden_size = 64
        self.rnn_hidden_size = 256

        self.core = nn.Sequential(
            nn.Linear(self.observation_size, self.core_hidden_size),
            nn.ReLU(True)
        )
        self.rnn = nn.GRU(self.core_hidden_size, self.rnn_hidden_size)

        self.policy = nn.Linear(self.rnn_hidden_size, self.action_size)
        self.value = nn.Linear(self.rnn_hidden_size, 1)

        self.rnn_state = None

    def reset_rnn(self):
        self.rnn_state = None

    def _forward(self, observation, rnn_state):
        x = self.core(observation)
        h, rnn_state = self.rnn(x, rnn_state)
        logits = self.policy(h[-1:])
        dist = Categorical(logits=logits)
        return dist, rnn_state

    def forward(self, observation):
        dist, self.rnn_state = self._forward(observation, self.rnn_state)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def act(self, observation):
        dist, self.rnn_state = self._forward(observation, self.rnn_state)
        action = dist.sample()
        return action.detach()

    def evaluate_actions(self, observations, actions, rnn_state=None):
        x = self.core(observations)
        h, _ = self.rnn(x, rnn_state)
        values = self.value(h).squeeze(-1)
        logits = self.policy(h[:-1])
        dist = Categorical(logits=logits)
        actions_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, actions_log_probs, entropy


#class Policy(nn.Module):
#    def __init__(self, observation_size, action_size, hidden_size=256, rnn_hidden_size=512, rnn_layers=2):
#        super(Policy, self).__init__()
#        self.observation_size = observation_size
#        self.action_size = action_size
#        self.hidden_size = hidden_size
#        self.rnn_hidden_size = rnn_hidden_size
#        self.rnn_layers = rnn_layers
#
#        self.core = nn.Sequential(
#            nn.Linear(observation_size, hidden_size),
#            nn.ReLU(True),
#        )
#        self.rnn = nn.GRU(hidden_size, rnn_hidden_size, rnn_layers)
#        self.rnn_state = None
#
#        self.policy = nn.Linear(rnn_hidden_size, action_size)
#        self.baseline = nn.Linear(rnn_hidden_size, 1)
#
#    def reset_rnn(self):
#        self.rnn_state = None
#
#    def forward(self, observation):
#        time_size, batch_size, _ = observation.shape
#
#        time_batch_merged_observation = torch.flatten(observation, 0, 1)
#
#        core_output = self.core(time_batch_merged_observation)
#        rnn_output, self.rnn_state = self.rnn(core_output.view(time_size, batch_size, -1), self.rnn_state)
#
#        time_batch_merged_rnn_output = torch.flatten(rnn_output, 0, 1)
#
#        policy_logits = self.policy(time_batch_merged_rnn_output)
#        baseline = self.baseline(time_batch_merged_rnn_output)
#
#        if self.training:
#            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
#            acton = action.clamp(0, self.action_size - 1)
#        else:
#            # Don't sample when testing.
#            action = torch.argmax(policy_logits, dim=1)
#
#        policy_logits = policy_logits.view(time_size, batch_size, self.action_size)
#        baseline = baseline.view(time_size, batch_size)
#        action = action.view(time_size, batch_size)
#
#        return policy_logits, baseline, action

def partial_load(model, path):
    old_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    old_dict = {k: v for k, v in old_dict.items() if k in model_dict and model_dict[k].shape == old_dict[k].shape}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)