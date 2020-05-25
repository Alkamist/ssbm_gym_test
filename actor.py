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
        self.opponent= None
        self.memory = None
        self.rnn_state = None

    def reset_rnn(self):
        self.rnn_state = torch.zeros(self.policy.rnn.num_layers, self.num_workers, self.policy.rnn.hidden_size, dtype=torch.float32, device=self.device)

    def initialize(self):
        if self.env is None:
            self.env = self.create_env_fn(self.rank)
        self.policy = Policy(self.env.observation_space.n, self.env.action_space.n).to(self.device)
        self.reset_rnn()
        self.memory = Memory()

    def performing(self):
        torch.manual_seed(self.seed + self.rank)

        self.initialize()
        observation = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while True:
                skip_batch = False

                self.policy.load_state_dict(self.shared_state_dict.state_dict())

                self.memory.rnn_states.append(self.rnn_state)

                for _ in range(self.episode_steps):
                    logits, baseline, action, self.rnn_state = self.policy(observation, self.rnn_state)

                    self.memory.observations.append(observation)
                    self.memory.actions.append(action)
                    self.memory.logits.append(logits)
                    self.memory.baselines.append(baseline)

                    step_env_with_timeout = timeout(5)(lambda : self.env.step(action[-1].cpu().numpy()))

                    try:
                        observation, reward, done, _ = step_env_with_timeout()

                        done = torch.tensor([done], dtype=torch.bool, device=self.device)
                        observation = torch.tensor([observation], dtype=torch.float32, device=self.device)
                        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

                        self.memory.rewards.append(reward)
                        self.memory.dones.append(done)

                        self.previous_action = action
                        self.previous_reward = reward

                    except:
                        skip_batch = True
                        observation = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)
                        self.reset_rnn()
                        break

                if skip_batch:
                    self.memory.clear_memory()
                else:
                    self.rollout_queue.put(self.memory.get_batch())

class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.baselines = []
        self.rnn_states = []

    def get_batch(self):
        observations = torch.cat(self.observations, dim=0).to('cpu')
        actions = torch.cat(self.actions, dim=0).to('cpu')
        rewards = torch.cat(self.rewards, dim=0).to('cpu')
        dones = torch.cat(self.dones, dim=0).to('cpu')
        logits = torch.cat(self.logits, dim=0).to('cpu')
        baselines = torch.cat(self.baselines, dim=0).to('cpu')
        rnn_states = torch.cat(self.rnn_states, dim=0).to('cpu')
        self.clear_memory()
        return (observations, actions, rewards, dones, logits, baselines, rnn_states)

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