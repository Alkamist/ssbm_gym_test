import torch

from melee_rollout_generator import MeleeRolloutGenerator


melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
    act_every=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actors = 8
episode_steps = 600
seed = 1


if __name__ == "__main__":
    melee_rollout_generator = MeleeRolloutGenerator(
        melee_options = melee_options,
        num_actors = num_actors,
        episode_steps = episode_steps,
        seed = seed,
        device = device
    )
    melee_rollout_generator.run()