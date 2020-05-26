import numpy as np
import torch

from melee_env import MeleeEnv
from rollout_generator import RolloutGenerator
from models import Policy

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
num_actors = 12
batch_size = 64
rollout_steps = 600
seed = 1


def create_melee_env(actor_id):
    return MeleeEnv(worker_id=actor_id, **melee_options)


if __name__ == "__main__":
    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.share_memory()

    melee_rollout_generator = RolloutGenerator(
        create_env_func = create_melee_env,
        num_actors = num_actors,
        rollout_steps = rollout_steps,
        shared_state_dict = shared_state_dict,
        seed = seed,
        device = device
    )

    #state_batch = []
    #action_batch = []
    #reward_batch = []
    #done_batch = []

    total_frames = 0
    while True:
        rollout = melee_rollout_generator.generate_rollout()

        #state_batch.append(rollout.states)
        #action_batch.append(rollout.actions)
        #reward_batch.append(rollout.rewards)
        #done_batch.append(rollout.dones)

        #if len(state_batch) >= batch_size:
        #    print("Learned")

        total_frames += len(rollout)
        print("Total Frames: %i" % total_frames)

    melee_rollout_generator.join_actor_processes()