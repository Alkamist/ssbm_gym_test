import time
from collections import deque
import numpy as np
import math
import random

from melee import Melee
from DQN import Agent

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

max_resets = 5
max_steps_before_reset = 105020

episode_length = 600
state_size = 10
action_size = 5

goal = [25.0, 0.0]
goal_embed = [goal[0] / 100.0, goal[1] / 100.0]

def calculate_reward(state_embed, goal_embed):
    def calculate_distance(x0, y0, x1, y1):
        return math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    max_distance_for_reward = 5.0
    x0 = state_embed[0]
    y0 = state_embed[1]
    x1 = goal_embed[0]
    y1 = goal_embed[1]
    return 1.0 if calculate_distance(x0, y0, x1, y1) < (max_distance_for_reward / 100.0) else 0.0

if __name__ == "__main__":
    agent = Agent(state_size=state_size, action_size=action_size)
    #agent.load("checkpoints/agent.pth")

    for reset_count in range(max_resets):
        melee = Melee(**options)
        state = melee.reset()

        rewards = deque(maxlen=3600)

        #HER_memory = []

        start_time = time.time()
        start_step = 0
        fps = 0
        for step_count in range(max_steps_before_reset):
            state_embed = melee.embed_state()
            action = agent.act(state_embed)
            #action = random.randrange(melee.num_actions)
            next_state = melee.step(action)
            next_state_embed = melee.embed_state()

            reward = calculate_reward(state_embed, goal_embed)
            done = (step_count > 0 and step_count % episode_length == 0) or (reward >= 1.0)

            agent.step(state_embed, action, reward, next_state_embed, done)

            #HER_memory.append([state_embed, action, 0.0, next_state_embed, done])
            #if done:
            #    # The episode failed.
            #    if reward <= 0.0:
            #        new_goal_embed = [state.players[0].x / 100.0, state.players[0].y / 100.0]
            #        for memory in HER_memory:
            #            agent.memory.add(memory[0] + new_goal_embed,
            #                            memory[1],
            #                            calculate_reward(memory[0], new_goal_embed),
            #                            memory[3] + new_goal_embed,
            #                            memory[4])
            #    HER_memory = []

            state = next_state
            rewards.append(reward)

            if step_count > 0 and step_count % 3600 == 0:
                #print("FPS:", fps, "Steps:", step_count, "R: %.4f" % np.mean(rewards), "X: %.2f" % state.players[0].x)
                print("FPS:", fps, "Steps:", step_count, "R: %.4f" % np.mean(rewards), "Epsilon: %.4f" % agent.epsilon)

            # Every 50000 steps save the model.
            if step_count > 0 and step_count % 50000 == 0:
                agent.save("checkpoints/" + str(reset_count) + "-" + str(step_count) + ".pth")

            current_time = time.time()
            if current_time - start_time >= 1.0:
                start_time = current_time
                fps = step_count - start_step
                start_step = step_count

        melee.close()