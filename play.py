import torch

from melee_env import MeleeEnv
from models import Policy

melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
    act_every=1,
)

if __name__ == "__main__":
    policy_net = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    #policy_net.load_state_dict(torch.load("checkpoints/agent.pth", map_location='cpu'))
    policy_net.eval()

    env = MeleeEnv(**melee_options)
    observation = env.reset()
    observation = torch.tensor([[observation]], dtype=torch.float32)

    while True:
        policy_logits, baseline, action = policy_net(observation)
        observation, reward, done, _ = env.step(action)
        observation = torch.tensor([[observation]], dtype=torch.float32)

        if reward != 0.0:
            print("Reward: %.4f" % reward)
