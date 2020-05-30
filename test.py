import torch

from melee_env import MeleeEnv
from DQN import DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

test_steps = 10800


if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device)
    network.load("checkpoints/agent.pth")
    network.evaluate()

    env = MeleeEnv(worker_id=1024, **melee_options)
    states = env.reset()
    states = torch.tensor([states], dtype=torch.float32, device=device)

    total_rewards = 0

    with torch.no_grad():
        for _ in range(test_steps):
            actions = network.act(states, epsilon=0.0)
            states, rewards, _, _ = env.step(actions.squeeze())
            states = torch.tensor([states], dtype=torch.float32, device=device)

            total_rewards += rewards[0]

    print(100.0 * total_rewards / test_steps)
