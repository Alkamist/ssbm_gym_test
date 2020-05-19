import math
import random
from collections import deque

import numpy as np

from melee import Melee
from replay_buffer import ReplayBuffer
from DQN import DQN

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

batch_size = 32
state_size = 792
action_size = 30

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
    reward = 1.0 if calculate_distance(x0, y0, x1, y1) < (max_distance_for_reward / 100.0) else -1.0
    return reward

if __name__ == "__main__":
    replay_buffer = ReplayBuffer(buffer_size=300000)
    replay_buffer.load("replay_memories/memories")
    dqn = DQN(state_size=state_size, action_size=action_size)
    #dqn.load("checkpoints/agent.pth")

    for reset_count in range(10):
        for learn_count in range(8000):
            dqn.learn(replay_buffer.sample(batch_size=batch_size), batch_size)

        rewards = deque(maxlen=7200)
        melee = Melee(**options)
        state = melee.reset()

        for step_count in range(7200):
            state_embed = melee.embed_state()
            action = dqn.act(state_embed, epsilon=0.0)
            next_state = melee.step(action)
            next_state_embed = melee.embed_state()
            reward = calculate_reward(state_embed, goal_embed)
            state = next_state
            rewards.append(reward)

        print("R: %.4f" % np.mean(rewards), "X: %.2f" % state.players[0].x)
        dqn.save("checkpoints/" + str(reset_count) + ".pth")

        melee.close()