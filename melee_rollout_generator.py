import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv


class MeleeActor(object):
    def __init__(self, melee_options, actor_id, episode_steps, rollout_queue, seed, device):
        self.melee_options = melee_options
        self.actor_id = actor_id
        self.episode_steps = episode_steps
        self.rollout_queue = rollout_queue
        self.seed = seed
        self.device = device
        self.env = None

    def run(self):
        if self.env is None:
            self.env = MeleeEnv(worker_id=self.actor_id, **self.melee_options)

        state = self.env.reset()

        with torch.no_grad():
            while True:
                try:
                    rollout = dict(
                        states = [],
                        actions = [],
                        rewards = [],
                        dones = [],
                    )

                    for _ in range(self.episode_steps):
                        action = self.env.action_space.sample()

                        rollout["states"].append(state)
                        rollout["actions"].append(action)

                        step_env_with_timeout = timeout(5)(lambda : self.env.step(action))
                        state, reward, done, _ = step_env_with_timeout()

                        rollout["rewards"].append(reward)
                        rollout["dones"].append(done)

                    self.rollout_queue.put(rollout)

                except KeyboardInterrupt:
                    self.env.close()
                except:
                    self.rollout_queue.put(dict(actor_crashed=self.actor_id))

class MeleeRolloutGenerator(object):
    def __init__(self, melee_options, num_actors, episode_steps, seed, device):
        self.is_initialized = False
        self.melee_options = melee_options
        self.num_actors = num_actors
        self.episode_steps = episode_steps
        self.seed = seed
        self.device = device

        self.rollout_queue = mp.Queue()

        self.actors = []
        self.actor_processes = []
        for actor_id in range(self.num_actors):
            self.create_new_actor(actor_id)

        self.is_initialized = True

    def create_new_actor(self, actor_id):
        actor = MeleeActor(
            melee_options = self.melee_options,
            actor_id = actor_id,
            episode_steps = self.episode_steps,
            rollout_queue = self.rollout_queue,
            seed = self.seed,
            device = self.device
        )
        actor_process = mp.Process(target=actor.run)

        if self.is_initialized:
            self.actor_processes[actor_id].terminate()
            self.actor_processes[actor_id] = actor_process
            self.actors[actor_id] = actor
        else:
            self.actor_processes.append(actor_process)
            self.actors.append(actor)

        self.actor_processes[actor_id].start()

    def run(self):
        total_frames = 0
        while True:
            rollout = self.rollout_queue.get()

            # Restart any crashed Dolphin instances.
            if "actor_crashed" in rollout.keys():
                crashed_actor_id = rollout["actor_crashed"]
                self.create_new_actor(crashed_actor_id)
            else:
                total_frames += len(rollout["states"])
                print("Total Frames: %i" % total_frames)

        for actor_process in self.actor_processes:
            actor_process.join()


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