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
        self.policy = Policy(self.env.action_space.n).to(self.device)
        self.memory = Memory()

    def performing(self):
        self.initialize()
        obs = self.env.reset()
        with torch.no_grad():
            while True:
                self.policy.load_state_dict(self.shared_state_dict.state_dict())

                try:
                    self.policy.reset_rnn()
                    obs = self.env.reset()
                except:
                    obs = obs[-1:]
                    print(obs.shape)

                self.memory.observations.append(obs)

                for _ in range(self.episode_steps):
                    action, action_log_prob = self.policy(obs)
                    self.memory.actions.append(action)
                    self.memory.actions_log_probs.append(action_log_prob)

                    send_action = action[-1].cpu().numpy()
                    obs, reward, done = self.env.step(send_action)
                    self.memory.observations.append(obs)
                    self.memory.rewards.append(torch.from_numpy(reward.astype(np.float32)))

                action, action_log_prob = self.policy(obs)
                self.memory.actions.append(action[0:-1])
                self.memory.actions_log_probs.append(action_log_prob[0:-1])

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

