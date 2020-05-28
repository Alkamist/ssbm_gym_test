import torch

from melee_env import MeleeEnv
from models import Policy

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
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions).to(device=device)
    #policy_net.load_state_dict(torch.load("checkpoints/agent.pth", map_location='cpu'))
    policy_net.eval()

    env = MeleeEnv(**melee_options)
    observations = env.reset()
    observations = torch.tensor([observations], dtype=torch.float32, device=device)

    with torch.no_grad():
        while True:
            #actions = [env.action_space.sample() for _ in range(2)]
            policy_logits, baselines, actions = policy_net(observations)
            observations, rewards, done, _ = env.step(actions.squeeze())
            observations = torch.tensor([observations], dtype=torch.float32, device=device)

            #if rewards[1] != 0.0:
            #    print("Reward: %.4f" % rewards[1])
