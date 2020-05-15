import numpy as np
import collections

#from melee_env import MeleeEnv
from test_env2 import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=False,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 9999999999

if __name__ == "__main__":
    agent = Agent(state_size=792, action_size=30)
    #agent.load("checkpoints/agent.pth")
    rewards = collections.deque(maxlen=1000)

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    for step_count in range(total_steps):
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        agent.step(observation, action, reward, next_observation, done)

        observation = next_observation

        rewards.append(reward)

        if step_count > 0 and step_count % 1000 == 0:
            avg_reward = np.mean(rewards)
            print('Step Count:', step_count, 'Average Reward: %.8f' % avg_reward)

        if step_count > 0 and step_count % 15000 == 0:
            agent.save("checkpoints/" + str(step_count) + ".pth")

    env.close()