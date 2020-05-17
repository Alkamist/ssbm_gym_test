import math

from melee import Melee
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

state_size = 2
action_size = 2

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
    agent = Agent(state_size=state_size + 2, action_size=action_size)
    agent.load("checkpoints/agent.pth")
    agent.evaluate()

    melee = Melee(**options)
    state = melee.reset()

    for step_count in range(9999999999):
        state_embed = melee.embed_state()
        action = agent.act(state_embed + goal_embed)
        reward = calculate_reward(state_embed, goal_embed)
        if reward >= 1.0:
            print(state.players[0].x)
        state = melee.step(action)

    melee.close()