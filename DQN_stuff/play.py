import math
import torch

from melee import Melee
from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
    worker_id=1,
)

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
    dqn = DQN(state_size=state_size, action_size=action_size)
    dqn.load("checkpoints/agent.pth")
    dqn.evaluate()

    melee = Melee(**options)
    state = melee.reset()

    for step_count in range(99999999999999):
        state_embed = torch.as_tensor([melee.embed_state()], device=device, dtype=torch.float32)
        action = dqn.act(state_embed, epsilon=0.01)
        next_state = melee.step(action)
        next_state_embed = torch.as_tensor([melee.embed_state()], device=device, dtype=torch.float32)
        reward = torch.as_tensor([[calculate_reward(state_embed[0], goal_embed)]], device=device, dtype=torch.float32)

        if reward[0].item() >= 1.0:
            print(reward[0].item())

    melee.close()