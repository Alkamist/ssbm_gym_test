import numpy as np
import collections

#from melee_env import MeleeEnv
from follow_env import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 100000

if __name__ == "__main__":
    agent = Agent(state_size=2, action_size=2)
    rewards = collections.deque(maxlen=1000)

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    for step_count in range(total_steps):
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        agent.step(observation, action, reward, next_observation, done)

        observation = next_observation

        rewards.append(reward)
        avg_reward = np.mean(rewards)

        if step_count > 0 and step_count % 1000 == 0:
            print('Step Count:', step_count, 'Average Reward: %.8f' % avg_reward)

        #if step_count > 0 and step_count % 10000 == 0:
        #    agent.save("checkpoints/" + str(step_count) + ".pth")

    env.close()