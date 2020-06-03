from copy import deepcopy

import torch

from melee_env import MeleeEnv
from DQN import DQN


melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, 1, device)
    network.load("checkpoints/agent.pth")

    env = MeleeEnv(**melee_options)
    states = env.reset()

    with torch.no_grad():
        while True:
            actions = []
            for player_id in range(2):
                state = torch.tensor(states[player_id], dtype=torch.float32, device=device).unsqueeze(0)
                actions.append(network.act(state, epsilon=0.0))

            next_states, rewards, dones, _ = env.step(actions)
            states = deepcopy(next_states)