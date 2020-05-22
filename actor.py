import numpy as np
import torch

from models import Policy

class Actor(object):
    def __init__(self, create_env_fn, episode_steps, rollout_queue, shared_state_dict, device):
        self.create_env_fn = create_env_fn
        self.episode_steps = episode_steps
        self.rollout_queue = rollout_queue
        self.shared_state_dict = shared_state_dict
        self.device = device
        self.env = None
        self.policy = None
        self.memory = None

    def initialize(self):
        if self.env is None:
            self.env = self.create_env_fn()
        self.policy = Policy(self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.memory = Memory()

    def performing(self):
        self.initialize()
        with torch.no_grad():
            while True:
                self.policy.load_state_dict(self.shared_state_dict.state_dict())

                #self.policy.reset_rnn()
                observation = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)

                for _ in range(self.episode_steps):
                    action, action_log_prob = self.policy(observation)

                    send_action = action[-1].cpu().numpy()
                    observation, reward, _, _ = self.env.step(send_action)
                    observation = torch.tensor([observation], dtype=torch.float32, device=self.device)
                    reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                    self.memory.actions.append(action)
                    self.memory.actions_log_probs.append(action_log_prob)
                    self.memory.observations.append(observation)
                    self.memory.rewards.append(reward)

#                action, action_log_prob = self.policy(obs)
#                self.memory.actions.append(action[0:-1])
#                self.memory.actions_log_probs.append(action_log_prob[0:-1])
#
                self.rollout_queue.put(self.memory.get_batch())

class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.observations = []
        self.actions = []
        self.actions_log_probs = []
        self.rewards = []

    def push(self, observation, action, action_log_prob, reward):
        self.observations.append(observation)
        self.actions.append(action)
        self.actions_log_probs.append(action_log_prob)
        self.rewards.append(torch.from_numpy(reward.astype(np.float32)))

    def get_batch(self):
        observations = torch.cat(self.observations, dim=0).to('cpu')
        actions = torch.cat(self.actions, dim=0).to('cpu')
        actions_log_probs = torch.cat(self.actions_log_probs, dim=0).to('cpu')
        rewards = torch.cat(self.rewards, dim=0).to('cpu')
        self.clear_memory()
        return (observations, actions, actions_log_probs, rewards)

    def __len__(self):
        return len(self.actions)

