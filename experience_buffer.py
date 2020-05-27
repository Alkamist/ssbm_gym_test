import torch

class ExperienceBuffer():
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.batch_size = batch_size
        self.num_traces = 0
        self.batch_is_ready = False
        self.states_batch = None
        self.actions_batch = None
        self.rewards_batch = None
        self.dones_batch = None
        self.logits_batch = None

    def add(self, states, actions, rewards, dones, logits):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.logits.append(logits)

        self.num_traces += len(states)

        if self.num_traces >= self.batch_size:
            self.num_traces -= self.batch_size

            self.states_batch, states_remain = torch.cat(self.states).split([self.batch_size, self.num_traces])
            self.actions_batch, actions_remain = torch.cat(self.actions).split([self.batch_size, self.num_traces])
            self.rewards_batch, rewards_remain = torch.cat(self.rewards).split([self.batch_size, self.num_traces])
            self.dones_batch, dones_remain = torch.cat(self.dones).split([self.batch_size, self.num_traces])
            self.logits_batch, logits_remain = torch.cat(self.logits).split([self.batch_size, self.num_traces])

            self.states = [states_remain]
            self.actions = [actions_remain]
            self.rewards = [rewards_remain]
            self.dones = [dones_remain]
            self.logits = [logits_remain]

            self.batch_is_ready = True

    def get_batch(self):
        states_batch = self.states_batch
        actions_batch = self.actions_batch
        rewards_batch = self.rewards_batch
        dones_batch = self.dones_batch
        logits_batch = self.logits_batch

        self.states_batch = None
        self.actions_batch = None
        self.rewards_batch = None
        self.dones_batch = None
        self.logits_batch = None

        self.batch_is_ready = False

        return states_batch, actions_batch, rewards_batch, dones_batch, logits_batch