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

state_size = 792
action_size = 30

if __name__ == "__main__":
    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load("checkpoints/agent.pth")
    agent.evaluate()

    env = MeleeEnv(max_episode_steps=9999999999, **options)
    observation = env.reset()

    for step_count in range(9999999999):
        action = agent.act(observation)
        observation, reward, done, info = env.step(env.action_space.from_index(action))

        #if reward != 0.0:
        #    print(reward)

    env.close()