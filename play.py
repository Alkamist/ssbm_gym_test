import numpy as np
import collections

from test_env2 import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 9999999999

if __name__ == "__main__":
    agent = Agent(state_size=792, action_size=30)
    agent.load("checkpoints/agent.pth")
    agent.evaluate()

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    for step_count in range(total_steps):
        action = agent.act(observation)
        observation, reward, done, info = env.step(env.action_space.from_index(action))

    env.close()