import timeit
import random
from copy import deepcopy

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from vectorized_env import VectorizedEnv
from learner import Learner
from models import Policy
from actor import Actor
from experience_buffer import ExperienceBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
    act_every=6,
)

num_actors = 2
workers_per_actor = 4
batch_size = 64
episode_steps = 50
seed = 2020
load_model = None
reset_policy = False

# Initialize shared memory used between the workers and
# the learner that contains the actor parameters.
def create_shared_state_dict():
    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)

    #if load_model is not None:
    #    partial_load(shared_state_dict, load_model)
    #    if reset_policy:
    #        shared_state_dict.policy.weight.data.zero_()
    #        shared_state_dict.policy.bias.data.zero_()

    shared_state_dict = shared_state_dict.share_memory()
    return shared_state_dict

def create_vectorized_env(actor_rank):
    def get_melee_env_fn(worker_id):
        return lambda : MeleeEnv(worker_id=worker_id + (actor_rank * workers_per_actor), **melee_options)
    return VectorizedEnv([get_melee_env_fn(worker_id) for worker_id in range(workers_per_actor)])

if __name__ == "__main__":
    processes = []

    experience_buffer = ExperienceBuffer(batch_size)
    p = mp.Process(target=experience_buffer.listening)
    p.start()
    processes.append(p)

    shared_state_dict = create_shared_state_dict()

    learner = Learner(
        observation_size=MeleeEnv.observation_size,
        num_actions=MeleeEnv.num_actions,
        lr=3e-5,
        c_hat=1.0,
        rho_hat=1.0,
        gamma=0.997,
        value_loss_coef=0.5,
        entropy_coef=0.0025,
        max_grad_norm=0.5,
        seed=seed,
        max_batch_repeat=3,
        episode_steps=episode_steps,
        queue_batch=experience_buffer.queue_batch,
        shared_state_dict=shared_state_dict,
        device=device,
    )

    actors = []
    for rank in range(num_actors):
        actors.append(Actor(
            create_env_fn=create_vectorized_env,
            episode_steps=episode_steps,
            seed=seed,
            rollout_queue=experience_buffer.queue_trace,
            shared_state_dict=shared_state_dict,
            device=device,
            rank=rank,
        ))

    for actor in actors:
        p = mp.Process(target=actor.performing)
        p.start()
        processes.append(p)

    learner.learning()

    for p in processes:
        p.join()