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
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

num_actors = 2
workers_per_actor = 2
batch_size = 128
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

def create_vectorized_env():
    return VectorizedEnv([lambda : MeleeEnv(**melee_options) for _ in range(workers_per_actor)])

if __name__ == "__main__":
    processes = []

    experience_buffer = ExperienceBuffer(batch_size)
    p = mp.Process(target=experience_buffer.listening)
    p.start()
    processes.append(p)

    shared_state_dict = create_shared_state_dict()

    learner = Learner(experience_buffer.queue_batch, shared_state_dict)

    actors = []
    for _ in range(num_actors):
        actors.append(Actor(
            create_env_fn=create_vectorized_env,
            episode_steps=600,
            rollout_queue=experience_buffer.queue_trace,
            shared_state_dict=shared_state_dict,
            device=device
        ))
    for actor in actors:
        p = mp.Process(target=actor.performing)
        p.start()
        processes.append(p)

    learner.learning()

    for p in processes:
        p.join()

#if __name__ == "__main__":
#    try:
#        env = VectorizedEnv([lambda : MeleeEnv(**melee_options) for _ in range(2)])
#        #env = MeleeEnv(**melee_options)
#        observation = env.reset()
#
#        #start_time = timeit.default_timer()
#        #fps = 0
#        while True:
#            action = [env.action_space.sample(), env.action_space.sample()]
#            observation, reward, done, _ = env.step(action)
#
#            #if reward != 0.0:
#            #    print(reward)
#
#            #fps += 1
#            #if timeit.default_timer() - start_time >= 1.0:
#            #    print("FPS: %.1f" % fps)
#            #    fps = 0
#            #    start_time = timeit.default_timer()
#    except KeyboardInterrupt:
#        env.close()