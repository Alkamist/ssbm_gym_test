#import numpy as np
#import collections

from test_env import MeleeEnv
#from DQN import Agent

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

total_steps = 20000

if __name__ == "__main__":
    #agent = Agent(state_size=4, action_size=5)
    #rewards = collections.deque(maxlen=1000)

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    for step_count in range(total_steps):
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)

        what_python_says = action.stick_MAIN.x * 2.0 - 1.0
        what_dolphin_says = env._game_state.players[0].controller.stick_MAIN.x
        if what_python_says != what_dolphin_says:
            print(step_count, what_python_says, what_dolphin_says)

        #print(what_python_says == what_dolphin_says, what_python_says, what_dolphin_says)

        #action = agent.act(observation)
        #next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        #agent.step(observation, action, reward, next_observation, done)

        #observation = next_observation

        #rewards.append(reward)

        #if step_count > 0 and step_count % 1000 == 0:
        #    avg_reward = np.mean(rewards)
        #    print('Step Count:', step_count, 'Average Reward: %.8f' % avg_reward)

    env.close()