import time
from collections import deque
import numpy as np

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

state_size = 792
action_size = 30

if __name__ == "__main__":
    agent = Agent(state_size=state_size, action_size=action_size)
    agent.load("checkpoints/agent.pth")

    env = MeleeEnv(max_episode_steps=999999999, **options)
    observation = env.reset()

    rewards = deque(maxlen=3600)

    start_time = time.time()
    start_step = 0
    fps = 0
    for step_count in range(9999999999):
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        agent.step(observation, action, reward, next_observation, done)

        observation = next_observation

        rewards.append(reward)

        if step_count > 0 and step_count % 3600 == 0:
            kills = 0
            deaths = 0
            for r in rewards:
                if r == 1.0:
                    kills += 1
                elif r == -1.0:
                    deaths += 1
            print("FPS:", fps, "Steps:", step_count, "KPM:", kills, "DPM:", deaths, "Score:", kills - deaths)

        # Every 15000 steps save the model.
        if step_count > 0 and step_count % 15000 == 0:
            agent.save("checkpoints/" + str(step_count) + ".pth")

        current_time = time.time()
        if current_time - start_time >= 1.0:
            start_time = current_time
            fps = step_count - start_step
            start_step = step_count

    env.close()