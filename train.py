import torch

from melee_env import MeleeEnv
from rollout_generator import RolloutGenerator


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actors = 1
rollout_steps = 20
seed = 1


def create_melee_env(actor_id):
    return MeleeEnv(worker_id=actor_id, **melee_options)


if __name__ == "__main__":
    melee_rollout_generator = RolloutGenerator(
        create_env_func = create_melee_env,
        num_actors = num_actors,
        rollout_steps = rollout_steps,
        seed = seed,
        device = device
    )

    total_frames = 0
    while True:
        rollout = melee_rollout_generator.generate_rollout()

        total_frames += len(rollout)
        print("Total Frames: %i" % total_frames)

    melee_rollout_generator.join_actor_processes()