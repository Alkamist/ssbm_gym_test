import torch

from models import Policy

class Actor(object):
    def __init__(self, create_env_fn, episode_steps, num_workers, seed, rollout_queue, shared_state_dict, rank, device):
        self.create_env_fn = create_env_fn
        self.episode_steps = episode_steps
        self.num_workers = num_workers
        self.seed = seed
        self.rollout_queue = rollout_queue
        self.shared_state_dict = shared_state_dict
        self.device = device
        self.rank = rank
        self.env = None
        self.policy = None
        self.memory = None

    def performing(self):
        torch.manual_seed(self.seed + self.rank)

        self.env = self.create_env_fn(self.rank)
        self.policy = Policy(self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.memory = Memory()

        observations = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while True:
                try:
                    self.policy.load_state_dict(self.shared_state_dict.state_dict())
                    self.policy.reset_rnn()

                    self.memory.observations.append(observations)

                    for _ in range(self.episode_steps):
                        actions, action_log_probs = self.policy(observations)

                        step_env_with_timeout = timeout(5)(lambda : self.env.step(actions[-1].cpu().numpy()))

                        observations, rewards, dones, _ = step_env_with_timeout()
                        observations = torch.tensor([observations], dtype=torch.float32, device=self.device)

                        self.memory.observations.append(observations)
                        self.memory.actions.append(actions)
                        self.memory.action_log_probs.append(action_log_probs)
                        self.memory.rewards.append(torch.tensor([rewards], dtype=torch.float32))
                        self.memory.dones.append(torch.tensor([dones], dtype=torch.bool))

                    actions, action_log_probs = self.policy(observations)
                    self.memory.actions.append(actions[0:-1])
                    self.memory.action_log_probs.append(action_log_probs[0:-1])

                    self.rollout_queue.put(self.memory.get_rollout())
                except KeyboardInterrupt:
                    self.env.close()
                except:
                    self.memory.clear_memory()
                    self.env.close()
                    for process in self.env.processes:
                        process.terminate()
                    self.env = self.create_env_fn(self.rank)
                    observation = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)

class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.observations = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []

    def get_rollout(self):
        observations = torch.cat(self.observations, dim=0).to('cpu')
        actions = torch.cat(self.actions, dim=0).to('cpu')
        action_log_probs = torch.cat(self.action_log_probs, dim=0).to('cpu')
        rewards = torch.cat(self.rewards, dim=0).to('cpu')
        dones = torch.cat(self.dones, dim=0).to('cpu')
        self.clear_memory()
        return (observations, actions, action_log_probs, rewards, dones)

    def __len__(self):
        return len(self.actions)

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