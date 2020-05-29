import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv

class Rollout(object):
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return self.rollout_steps

class MeleeActor(object):
    def __init__(self, actor_id, rollout_steps, rollout_queue, seed, device, dolphin_options):
        self.actor_id = actor_id
        self.rollout_steps = rollout_steps
        self.rollout_queue = rollout_queue
        self.seed = seed
        self.device = device
        self.dolphin_options = dolphin_options
        self.env = None

    def run(self):
        self.env = MeleeEnv(worker_id=self.actor_id, **self.dolphin_options)

        states = self.env.reset()

        with torch.no_grad():
            while True:
                try:
                    rollout = Rollout(self.rollout_steps)

                    for _ in range(self.rollout_steps):
                        actions = [self.env.action_space.sample() for _ in range(2)]

                        rollout.states.append(states)
                        rollout.actions.append(actions)

                        step_env_with_timeout = timeout(5)(lambda : self.env.step(actions))
                        states, rewards, dones, _ = step_env_with_timeout()

                        rollout.rewards.append(rewards)
                        rollout.dones.append(dones)

                    self.rollout_queue.put(rollout)

                except KeyboardInterrupt:
                    self.env.close()
                except:
                    self.rollout_queue.put(self.actor_id)

class MeleeRolloutGenerator(object):
    def __init__(self, num_actors, rollout_steps, seed, device, dolphin_options):
        self.num_actors = num_actors
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.device = device
        self.dolphin_options = dolphin_options

        self.rollout_queue = mp.Queue()

        self.actors = [None] * self.num_actors
        self.actor_processes = [None] * self.num_actors
        for actor_id in range(self.num_actors):
            self.create_new_actor(actor_id)

    def create_new_actor(self, actor_id):
        actor = MeleeActor(
            actor_id = actor_id,
            rollout_steps = self.rollout_steps,
            rollout_queue = self.rollout_queue,
            seed = self.seed,
            device = self.device,
            dolphin_options = self.dolphin_options
        )
        actor_process = mp.Process(target=actor.run)

        if self.actor_processes[actor_id] is not None:
            self.actor_processes[actor_id].terminate()

        self.actor_processes[actor_id] = actor_process
        self.actors[actor_id] = actor
        self.actor_processes[actor_id].start()

    def join_actor_processes(self):
        for actor_process in self.actor_processes:
            actor_process.join()

    def generate_rollout(self):
        rollout = self.rollout_queue.get()

        # Restart any crashed Dolphin instances.
        while not isinstance(rollout, Rollout):
            crashed_actor_id = rollout
            self.create_new_actor(crashed_actor_id)
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







#import torch
#import torch.multiprocessing as mp
#
#from melee_env import MeleeEnv
#
#def _run_melee(output_queue, actor_id, dolphin_options):
#    env = MeleeEnv(worker_id=actor_id, **dolphin_options)
#
#    observation = env.reset()
#    while True:
#        actions = [env.action_space.sample() for _ in range(2)]
#        observations, rewards, done, _ = env.step(actions)
#        output_queue.put(1)
#
#class MeleeRolloutGenerator():
#    def __init__(self, num_actors, workers_per_actor, dolphin_options):
#        self.num_actors = num_actors
#        self.workers_per_actor = workers_per_actor
#        self.dolphin_options = dolphin_options
#        self.melee_processes = [None] * self.num_actors
#        self.output_queue = mp.Queue
#
#    def start(self):
#        for actor_id in range(self.num_actors):
#            self.start_melee_process(actor_id)
#
#
#
#    def start_melee_process(self, actor_id):
#        p = mp.Process(target=_run_melee, args=(self.output_queue, actor_id, self.dolphin_options))
#        p.start()
#        self.melee_processes[actor_id] = p






#if __name__ == "__main__":
#    output_queue = mp.Queue()
#
#    melee_processes = []
#    for actor_id in range(num_actors):
#        p = mp.Process(target=run_melee, args=(actor_id, output_queue))
#        p.start()
#        melee_processes.append(p)
#
#    total_frames = 0
#    while True:
#        frames = output_queue.get()
#        total_frames += frames
#
#        if total_frames % 3600 == 0:
#            print(total_frames)
#
#    for p in melee_processes:
#        p.join()










#def call_env_function(env_function, output, args):
#    if args is None:
#        output.append(env_function())
#    else:
#        output.append(env_function(args))
#
#def threaded_env_function_call(envs, function_name, function_args_list=None):
#    output = []
#    threads = []
#
#    for actor_id, env in enumerate(envs):
#        env_function = getattr(env, function_name)
#        function_args = function_args_list[actor_id] if isinstance(function_args_list, list) else None
#        t = threading.Thread(target=call_env_function, args=(env_function, output, function_args))
#        t.start()
#        threads.append(t)
#
#    for t in threads:
#        t.join()
#
#    return output
#
#if __name__ == "__main__":
#    melee_envs = [MeleeEnv(worker_id=actor_id, **dolphin_options) for actor_id in range(num_actors)]
#
#    observations = threaded_env_function_call(melee_envs, "reset")
#
#    frames_since_last_print = 0
#    t = time.perf_counter()
#    while True:
#        actions = [[random.randrange(MeleeEnv.num_actions) for _ in range(2)] for _ in range(num_actors)]
#
#        outputs = threaded_env_function_call(melee_envs, "step", actions)
#        #observations, rewards, done, _ = env.step(actions)
#
#        frames_since_last_print += num_actors
#
#        t_ = time.perf_counter()
#        delta_t = t_ - t
#        if delta_t >= 1.0:
#            print(frames_since_last_print)
#            frames_since_last_print = 0
#            t = t_