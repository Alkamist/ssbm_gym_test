import torch
from torch import nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=256, rnn_hidden_size=512, rnn_layers=1):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        self.core = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(True),
        )
        self.rnn = nn.GRU(hidden_size, rnn_hidden_size, rnn_layers)

        self.policy = nn.Linear(rnn_hidden_size, action_size)
        self.baseline = nn.Linear(rnn_hidden_size, 1)

    def forward(self, observation, rnn_state=None):
        time_size, batch_size, _ = observation.shape

        time_batch_merged_observation = torch.flatten(observation, 0, 1)

        core_output = self.core(time_batch_merged_observation)
        rnn_output, rnn_state = self.rnn(core_output.view(time_size, batch_size, -1), rnn_state)

        time_batch_merged_rnn_output = torch.flatten(rnn_output, 0, 1)

        policy_logits = self.policy(time_batch_merged_rnn_output)
        baseline = self.baseline(time_batch_merged_rnn_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(time_size, batch_size, self.action_size)
        baseline = baseline.view(time_size, batch_size)
        action = action.view(time_size, batch_size)

        return policy_logits, baseline, action, rnn_state

def partial_load(model, path):
    old_dict = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()
    old_dict = {k: v for k, v in old_dict.items() if k in model_dict and model_dict[k].shape == old_dict[k].shape}
    model_dict.update(old_dict)
    model.load_state_dict(model_dict)