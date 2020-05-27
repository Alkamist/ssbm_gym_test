import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from rollout_generator import RolloutGenerator


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
rollout_steps = 600
seed = 1


def get_melee_env_func(actor_id):
    return lambda : MeleeEnv(worker_id=actor_id, **melee_options)


if __name__ == "__main__":
    rollout_queue = mp.Queue()

    rollout_generator = RolloutGenerator(
        env_func = get_melee_env_func,
        num_actors = num_actors,
        rollout_steps = rollout_steps,
        rollout_queue = rollout_queue,
        seed = seed,
        device = device
    )

    process = mp.Process(target=rollout_generator.run)
    process.start()

    total_steps = 0
    while True:
        rollout = rollout_queue.get()

        total_steps += len(rollout) * num_actors
        print(total_steps)

    process.join()