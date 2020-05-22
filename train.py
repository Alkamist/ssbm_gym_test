import timeit
import random
from copy import deepcopy

import torch

from melee_env import MeleeEnv
#from model import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

if __name__ == "__main__":
    env = MeleeEnv(**melee_options)
    observation = env.reset()

    #net = Net(melee.state_size, melee.num_actions).to(device)
    #net.eval()

    start_time = timeit.default_timer()
    fps = 0
    while True:
        #state_tensor = melee_state_to_tensor(state, device)
        #action = net(state_tensor).max(2)[1].squeeze().item()

        action = random.randrange(30)
        observation, reward, done, _ = env.step(action)

        if reward != 0.0:
            print(reward)

        fps += 1
        if timeit.default_timer() - start_time >= 1.0:
            print("FPS: %.1f" % fps)
            fps = 0
            start_time = timeit.default_timer()