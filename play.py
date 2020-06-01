import random

import torch

from melee_env import MeleeEnv
from DQN import Policy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)


if __name__ == "__main__":
    policy = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device)
    policy.load_state_dict(torch.load("checkpoints/agent.pth"))
    policy.eval()

    env = MeleeEnv(worker_id=512, **melee_options)
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device)

    frames = 0
    with torch.no_grad():
        while True:
            actions = policy(states).max(1)[1]
            states, rewards, _, _ = env.step(actions.squeeze().cpu().numpy())
            states = torch.tensor(states, dtype=torch.float32, device=device)

            #if frames % 20 == 0:
            #    print(states)

            frames += 1

            if rewards[1] != 0.0:
                print("Reward: %.4f" % rewards[1])
