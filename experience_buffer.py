import torch
from torch.multiprocessing import Queue

class ExperienceBuffer():
    def __init__(self, batch_size):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []

        self.queue_trace = Queue(maxsize=30)
        self.queue_batch = Queue(maxsize=3)
        self.batch_size = batch_size
        self.num_traces = 0

    def listening(self):
        while True:
            trace = self.queue_trace.get(block=True)

            self.observations.append(trace[0])
            self.actions.append(trace[1])
            self.rewards.append(trace[2])
            self.dones.append(trace[3])
            self.logits.append(trace[4])

            self.num_traces += trace[0].shape[1]

            if self.num_traces >= self.batch_size:
                self.num_traces -= self.batch_size

                observations_batch, observations_remain = torch.cat(self.observations, dim=1).split([self.batch_size, self.num_traces], dim=1)
                actions_batch, actions_remain = torch.cat(self.actions, dim=1).split([self.batch_size, self.num_traces], dim=1)
                rewards_batch, rewards_remain = torch.cat(self.rewards, dim=1).split([self.batch_size, self.num_traces], dim=1)
                dones_batch, dones_remain = torch.cat(self.dones, dim=1).split([self.batch_size, self.num_traces], dim=1)
                logits_batch, logits_remain = torch.cat(self.logits, dim=1).split([self.batch_size, self.num_traces], dim=1)

                self.observations = [observations_remain]
                self.actions = [actions_remain]
                self.rewards = [rewards_remain]
                self.dones = [dones_remain]
                self.logits = [logits_remain]

                self.queue_batch.put((
                    observations_batch,
                    actions_batch,
                    rewards_batch,
                    dones_batch,
                    logits_batch
                ))