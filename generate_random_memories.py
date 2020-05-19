import time
import math
import random

from melee import Melee
from replay_buffer import ReplayBuffer

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

max_resets = 3
max_steps_before_reset = 100000

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

def record_random_memories(step_count, melee, replay_buffer):
    state_embed = melee.embed_state()
    action = random.randrange(melee.num_actions)
    next_state = melee.step(action)
    next_state_embed = melee.embed_state()
    reward = calculate_reward(state_embed, goal_embed)
    replay_buffer.add(state_embed, action, next_state_embed, reward)
    if step_count % 20000 == 0:
        replay_buffer.save("replay_memories/memories")
    return next_state, reward

if __name__ == "__main__":
    replay_buffer = ReplayBuffer(buffer_size=300000)

    for reset_count in range(max_resets):
        melee = Melee(**options)
        state = melee.reset()

        start_time = time.time()
        start_step = 0
        fps = 0
        for step_count in range(max_steps_before_reset):
            state_embed = melee.embed_state()
            action = random.randrange(melee.num_actions)
            next_state = melee.step(action)
            next_state_embed = melee.embed_state()
            reward = calculate_reward(state_embed, goal_embed)
            replay_buffer.add(state_embed, action, next_state_embed, reward)
            if step_count % 20000 == 0:
                replay_buffer.save("replay_memories/memories")

            state = next_state
            if step_count > 0 and step_count % 3600 == 0:
                print("FPS:", fps, "Steps:", step_count)

            current_time = time.time()
            if current_time - start_time >= 1.0:
                start_time = current_time
                fps = step_count - start_step
                start_step = step_count

        melee.close()