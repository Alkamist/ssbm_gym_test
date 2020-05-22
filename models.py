import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def _forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)

    def forward(self, observation):
        dist = self._forward(observation)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def act(self, observation):
        dist = self._forward(observation)
        action = dist.sample()
        return action.detach()

#import torch
#from torch import nn
#import torch.nn.functional as F
#from torch.distributions import Categorical
#
#class MeleeEmbedding(nn.Module):
#    def __init__(self, state_size=24, embedding_size=32):
#        super(MeleeEmbedding, self).__init__()
#        self.actionstate_embedding = nn.Embedding(400, embedding_size)
#        self.actionstate_embedding.weight.data.zero_()
#        self.output_size = 2 * (embedding_size + int(state_size / 2) - 1)
#
#    def embed(self, x):
#        a, s = x.split([1, x.shape[-1] - 1], dim=-1)
#        actionstate = self.actionstate_embedding(a.long().clamp(0, 399)).squeeze(-2)
#        return torch.cat([actionstate, s], dim=-1)
#
#    def forward(self, x):
#        agent, opponent = x.chunk(2, dim=-1)
#        agent = self.embed(agent)
#        opponent = self.embed(opponent)
#        return torch.cat([agent, opponent], dim=-1)
#
#class Policy(nn.Module):
#    def __init__(self, input_size, output_size, hidden_size=512):
#        super(Policy, self).__init__()
#        self.melee_embedding = MeleeEmbedding(state_size=input_size)
#        self.fc1 = nn.Linear(self.melee_embedding.output_size, hidden_size)
#        self.fc2 = nn.Linear(hidden_size, hidden_size)
#        self.fc3 = nn.Linear(hidden_size, output_size)
#
#    def forward(self, x):
#        x = self.melee_embedding(x)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        return self.fc3(x)