import torch

from melee_env import MeleeEnv
from vectorized_env import VectorizedEnv
from timeout import timeout
from models import Policy


class Rollout(object):
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []

    def __len__(self):
        return self.rollout_steps

class RolloutGenerator(object):
    def __init__(self, env_func, num_actors, rollout_steps, rollout_queue, shared_state_dict, seed, device):
        self.env_func = env_func
        self.num_actors = num_actors
        self.rollout_steps = rollout_steps
        self.rollout_queue = rollout_queue
        self.shared_state_dict = shared_state_dict
        self.seed = seed
        self.device = device
        self.policy = None
        self.env = None

    def run(self):
        torch.manual_seed(self.seed)

        self.env = VectorizedEnv([self.env_func(actor_id) for actor_id in range(self.num_actors)])
        self.policy = Policy(self.env.observation_space.n, self.env.action_space.n).to(self.device)

        states = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            while True:
                try:
                    self.policy.load_state_dict(self.shared_state_dict.state_dict())

                    rollout = Rollout(self.rollout_steps)

                    for _ in range(self.rollout_steps):
                        logits, _, actions = self.policy(states)

                        rollout.states.append(states.cpu())

                        step_env_with_timeout = timeout(5)(lambda : self.env.step(actions.squeeze().cpu()))
                        states, rewards, dones, _ = step_env_with_timeout()

                        states = torch.tensor([states], dtype=torch.float32, device=self.device)
                        rewards = torch.tensor([[rewards]], dtype=torch.float32)
                        dones = torch.tensor([[dones]], dtype=torch.bool)

                        rollout.actions.append(actions.cpu())
                        rollout.rewards.append(rewards)
                        rollout.dones.append(dones)
                        rollout.logits.append(logits.cpu())

                    self.rollout_queue.put(rollout)

                except KeyboardInterrupt:
                    self.env.close()
                except:
                    self.env.close()
                    for process in self.env.processes:
                        process.terminate()
                    self.env = VectorizedEnv([self.env_func(actor_id) for actor_id in range(self.num_actors)])
                    states = torch.tensor([self.env.reset()], dtype=torch.float32, device=self.device)