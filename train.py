import time
import threading
import random

#import torch
#import torch.multiprocessing as mp

from melee_env import MeleeEnv

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

num_actors = 16

def call_env_function(env_function, output, args):
    if args is None:
        output.append(env_function())
    else:
        output.append(env_function(args))

def threaded_env_function_call(envs, function_name, function_args_list=None):
    output = []
    threads = []

    for actor_id, env in enumerate(envs):
        env_function = getattr(env, function_name)
        function_args = function_args_list[actor_id] if isinstance(function_args_list, list) else None
        t = threading.Thread(target=call_env_function, args=(env_function, output, function_args))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return output

if __name__ == "__main__":
    melee_envs = [MeleeEnv(worker_id=actor_id, **melee_options) for actor_id in range(num_actors)]

    observations = threaded_env_function_call(melee_envs, "reset")

    frames_since_last_print = 0
    t = time.perf_counter()
    while True:
        actions = [[random.randrange(MeleeEnv.num_actions) for _ in range(2)] for _ in range(num_actors)]

        outputs = threaded_env_function_call(melee_envs, "step", actions)
        #observations, rewards, done, _ = env.step(actions)

        frames_since_last_print += num_actors

        t_ = time.perf_counter()
        delta_t = t_ - t
        if delta_t >= 1.0:
            print(frames_since_last_print)
            frames_since_last_print = 0
            t = t_