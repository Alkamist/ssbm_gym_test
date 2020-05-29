import torch

from melee_rollout_generator import MeleeRolloutGenerator

melee_options = dict(
    render=True,
    speed=1,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actor_pools = 2
num_actors_per_pool = 2
rollout_steps = 600
seed = 1

if __name__ == "__main__":
    melee_rollout_generator = MeleeRolloutGenerator(
        num_actor_pools=num_actor_pools,
        num_actors_per_pool=num_actors_per_pool,
        rollout_steps=rollout_steps,
        seed=seed,
        device=device,
        dolphin_options=melee_options,
    )

    total_frames = 0
    while True:
        rollout = melee_rollout_generator.generate_rollout()

        total_frames += len(rollout) * num_actors_per_pool
        print(total_frames)

    melee_rollout_generator.join_actor_pool_processes()