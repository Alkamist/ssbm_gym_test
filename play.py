import torch

from melee_env import MeleeEnv
from DQN import DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

if __name__ == "__main__":
    network = DQN(MeleeEnv.observation_size, MeleeEnv.num_actions, device)
    network.load("checkpoints/agent.pth")
    network.evaluate()

    env = MeleeEnv(worker_id=512, **melee_options)
    states = env.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device)

    with torch.no_grad():
        while True:
            actions = network.act(states, epsilon=0.0)
            states, rewards, _, _ = env.step(actions.squeeze())
            states = torch.tensor(states, dtype=torch.float32, device=device)

            #if rewards[1] != 0.0:
            #    print("Reward: %.4f" % rewards[0])
