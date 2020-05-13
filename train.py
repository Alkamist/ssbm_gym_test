import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from melee_env import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=False,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 100000

if __name__ == "__main__":
    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, lr=0.003, n_actions=env.action_space.n, input_dims=[env.observation_space.n])

    rewards = collections.deque(maxlen=100)

    for step_count in range(total_steps):
        action = agent.choose_action(observation.data)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        agent.store_transition(observation.data, action, reward, next_observation.data, done)

        agent.learn()
        observation = next_observation

        rewards.append(reward)
        avg_reward = np.mean(rewards)

        if step_count % 1000 == 0:
            print('Step Count: ', step_count, 'Avg Reward: ', avg_reward)

    env.close()