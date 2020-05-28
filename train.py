from copy import deepcopy

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from vectorized_env import VectorizedEnv
from learner import Learner
from models import Policy, partial_load
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
    stage='final_destination',
)

num_actors = 4
workers_per_actor = 4
batch_size = 64
episode_steps = 600
seed = 1
#load_model = "checkpoints/agent.pth"
load_model = None
reset_policy = False

def partial_load_model_to_state_dict(state_dict):
    if load_model is not None:
        partial_load(state_dict, load_model)
        if reset_policy:
            state_dict.policy.weight.data.zero_()
            state_dict.policy.bias.data.zero_()

def create_vectorized_env(actor_rank):
    def get_melee_env_fn(worker_id):
        global melee_options
        unique_id = worker_id + (actor_rank * workers_per_actor)
        modified_melee_options = deepcopy(melee_options)
        #if unique_id == 1:
        #    modified_melee_options["render"] = True
        return lambda : MeleeEnv(worker_id=unique_id, **modified_melee_options)
    return VectorizedEnv([get_melee_env_fn(worker_id) for worker_id in range(workers_per_actor)])

if __name__ == "__main__":
    processes = []

    experience_buffer = ExperienceBuffer(batch_size)
    p = mp.Process(target=experience_buffer.listening)
    p.start()
    processes.append(p)

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    partial_load_model_to_state_dict(shared_state_dict)
    shared_state_dict.share_memory()

    learner = Learner(
        observation_size=MeleeEnv.observation_size,
        num_actions=MeleeEnv.num_actions,
        lr=3e-5,
        discounting=0.99,
        baseline_cost=0.5,
        entropy_cost=0.0025,
        grad_norm_clipping=40.0,
        save_interval=1,
        seed=seed,
        queue_batch=experience_buffer.queue_batch,
        shared_state_dict=shared_state_dict,
        device=device,
    )
    partial_load_model_to_state_dict(learner.policy)

    actors = []
    for rank in range(num_actors):
        actors.append(Actor(
            create_env_fn=create_vectorized_env,
            episode_steps=episode_steps,
            num_workers=workers_per_actor,
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