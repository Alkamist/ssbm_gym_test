import torch
import torch.multiprocessing as mp

from models import Policy

class Rollout(object):
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.baselines = []

    def __len__(self):
        return self.rollout_steps

class Actor(object):
    def __init__(self, create_env_func, actor_id, rollout_steps, rollout_queue, shared_state_dict, seed, device):
        self.create_env_func = create_env_func
        self.actor_id = actor_id
        self.rollout_steps = rollout_steps
        self.rollout_queue = rollout_queue
        self.shared_state_dict = shared_state_dict
        self.seed = seed
        self.device = device
        self.policy = None
        self.env = None

    def run(self):
        torch.manual_seed(self.seed + self.actor_id)

        if self.env is None:
            self.env = self.create_env_func(self.actor_id)

        self.policy = Policy(self.env.observation_space.n, self.env.action_space.n).to(device=self.device)

        state = torch.tensor([[self.env.reset()]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while True:
                #try:
                self.policy.load_state_dict(self.shared_state_dict.state_dict())

                rollout = Rollout(self.rollout_steps)

                for _ in range(self.rollout_steps):
                    logits, baseline, action = self.policy(state)

                    rollout.states.append(1)
                    #rollout.states.append(state.squeeze())

                    step_env_with_timeout = timeout(5)(lambda : self.env.step(action.squeeze().cpu()))
                    state, reward, done, _ = step_env_with_timeout()

                    state = torch.tensor([[state]], dtype=torch.float32, device=self.device)
                    #reward = reward
                    #done = done

                    #rollout.actions.append(action)
                    #rollout.rewards.append(reward)
                    #rollout.dones.append(reward)
                    #rollout.logits.append(logits.squeeze())
                    #rollout.baselines.append(baseline.squeeze())

                self.rollout_queue.put(rollout)

                #except KeyboardInterrupt:
                #    self.env.close()
                #except:
                #    self.rollout_queue.put(self.actor_id)

class RolloutGenerator(object):
    def __init__(self, create_env_func, num_actors, rollout_steps, shared_state_dict, seed, device):
        self.create_env_func = create_env_func
        self.is_initialized = False
        self.num_actors = num_actors
        self.rollout_steps = rollout_steps
        self.shared_state_dict = shared_state_dict
        self.seed = seed
        self.device = device

        self.rollout_queue = mp.Queue()

        self.actors = []
        self.actor_processes = []
        for actor_id in range(self.num_actors):
            self.create_new_actor(actor_id)

        self.is_initialized = True

    def create_new_actor(self, actor_id):
        actor = Actor(
            create_env_func = self.create_env_func,
            actor_id = actor_id,
            rollout_steps = self.rollout_steps,
            rollout_queue = self.rollout_queue,
            shared_state_dict = self.shared_state_dict,
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