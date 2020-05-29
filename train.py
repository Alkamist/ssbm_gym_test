import torch

from melee_rollout_generator import MeleeRolloutGenerator

melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actors = 4
rollout_steps = 600
seed = 1

if __name__ == "__main__":
    melee_rollout_generator = MeleeRolloutGenerator(
        num_actors=num_actors,
        rollout_steps=rollout_steps,
        seed=seed,
        device=device,
        dolphin_options=melee_options,
    )

    total_frames = 0
    while True:
        rollout = melee_rollout_generator.generate_rollout()

        total_frames += len(rollout)
        print(total_frames)

    melee_rollout_generator.join_actor_processes()