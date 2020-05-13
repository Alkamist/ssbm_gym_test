#from melee_env import MeleeEnv
from follow_env import MeleeEnv
from DQN import Agent

import time

options = dict(
    windows=True,
    render=False,
    speed=0,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 100000

if __name__ == "__main__":
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, lr=0.001, n_actions=3, input_dims=2)

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    for step_count in range(total_steps):
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))
        #action = env.action_space.sample()
        #next_observation, reward, done, info = env.step(action)

        agent.store_transition(observation, action, reward, next_observation, done)

        agent.learn()
        observation = next_observation

        if step_count > 0 and step_count % 1000 == 0:
            print('Step Count:', step_count, 'Reward: %.8f' % reward)

        #if step_count > 0 and step_count % 10000 == 0:
        #    agent.save("checkpoints/" + str(step_count) + ".pth")

    env.close()