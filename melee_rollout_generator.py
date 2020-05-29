import random
import threading

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv

def call_env_function(env_function, output, args):
    if args is None:
        output.append(env_function())
    else:
        output.append(env_function(args))

def threaded_env_function_call(envs, function_name, function_args_list=None):
    output = []
    threads = []

    for pool_id, env in enumerate(envs):
        env_function = getattr(env, function_name)
        function_args = function_args_list[pool_id] if isinstance(function_args_list, list) else None
        t = threading.Thread(target=call_env_function, args=(env_function, output, function_args))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return output

class Rollout(object):
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return self.rollout_steps

class SynchronousMeleeActorPool(object):
    def __init__(self, pool_id, num_actors, rollout_steps, rollout_queue, seed, device, dolphin_options):
        self.pool_id = pool_id
        self.num_actors = num_actors
        self.rollout_steps = rollout_steps
        self.rollout_queue = rollout_queue
        self.seed = seed
        self.device = device
        self.dolphin_options = dolphin_options
        self.num_ai_players = 2
        self.melee_envs = None

    def run(self):
        self.melee_envs = [MeleeEnv(worker_id=(self.pool_id * self.num_actors) + actor_id, **self.dolphin_options) for actor_id in range(self.num_actors)]

        states = threaded_env_function_call(self.melee_envs, "reset")

        with torch.no_grad():
            while True:
                try:
                    rollout = Rollout(self.rollout_steps)

                    for _ in range(self.rollout_steps):
                        actions = [[random.randrange(MeleeEnv.num_actions) for _ in range(self.num_ai_players)] for _ in range(self.num_actors)]

                        rollout.states.append(states)
                        rollout.actions.append(actions)

                        step_env_with_timeout = timeout(5)(lambda : threaded_env_function_call(self.melee_envs, "step", actions))
                        outputs = step_env_with_timeout()

                        states, rewards, dones = [], [], []
                        for actor_id in range(self.num_actors):
                            state, reward, done, _ = outputs[actor_id]
                            states.append(state)
                            rewards.append(reward)
                            dones.append(done)

                        rollout.rewards.append(rewards)
                        rollout.dones.append(dones)

                    self.rollout_queue.put(rollout)

                except KeyboardInterrupt:
                    for env in self.melee_envs:
                        env.close()

                except:
                    for env in self.melee_envs:
                        env.close()
                    self.rollout_queue.put(self.pool_id)

class MeleeRolloutGenerator(object):
    def __init__(self, num_actor_pools, num_actors_per_pool, rollout_steps, seed, device, dolphin_options):
        self.num_actor_pools = num_actor_pools
        self.num_actors_per_pool = num_actors_per_pool
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.device = device
        self.dolphin_options = dolphin_options

        self.rollout_queue = mp.Queue()

        self.actor_pools = [None] * self.num_actor_pools
        self.actor_pool_processes = [None] * self.num_actor_pools

        for pool_id in range(self.num_actor_pools):
            self.create_new_actor_pool(pool_id)

    def create_new_actor_pool(self, pool_id):
        actor = SynchronousMeleeActorPool(
            pool_id = pool_id,
            num_actors = self.num_actors_per_pool,
            rollout_steps = self.rollout_steps,
            rollout_queue = self.rollout_queue,
            seed = self.seed,
            device = self.device,
            dolphin_options = self.dolphin_options
        )
        actor_pool_process = mp.Process(target=actor.run)

        if self.actor_pool_processes[pool_id] is not None:
            self.actor_pool_processes[pool_id].terminate()

        self.actor_pool_processes[pool_id] = actor_pool_process
        self.actor_pools[pool_id] = actor
        self.actor_pool_processes[pool_id].start()

    def join_actor_pool_processes(self):
        for actor_pool_process in self.actor_pool_processes:
            actor_pool_process.join()

    def generate_rollout(self):
        rollout = self.rollout_queue.get()

        # Restart any crashed actor pools.
        while not isinstance(rollout, Rollout):
            crashed_actor_pool_id = rollout
            self.create_new_actor_pool(crashed_actor_pool_id)
            rollout = self.rollout_queue.get()

        return rollout

from threading import Thread
import functools

def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, seconds_before_timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print('error starting thread')
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco