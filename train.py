import timeit
import random
from copy import deepcopy

import torch

from melee import Melee, melee_state_to_tensor
from model import Net

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
)

def calculate_reward(state, next_state):
    return 1.0 if abs(next_state.players[0].x - 25.0) < 5.0 else 0.0

if __name__ == "__main__":
    melee = Melee(**options)
    state = melee.reset()

    net = Net(melee.state_size, melee.num_actions).to(device)
    net.eval()

    start_time = timeit.default_timer()
    fps = 0
    while True:
        state_tensor = melee_state_to_tensor(state, device)
        #action = net(state_tensor).max(2)[1].squeeze().item()

        action = random.randrange(melee.num_actions)
        next_state = melee.step(action)
        reward = calculate_reward(state, next_state)

        if reward != 0.0:
            print(reward)

        state = deepcopy(next_state)

        fps += 1
        if timeit.default_timer() - start_time >= 1.0:
            #print("FPS: %.1f" % fps)
            fps = 0
            start_time = timeit.default_timer()
