from melee_env import MeleeEnv
from vectorized_env import VectorizedEnv
from timeout import timeout


class Rollout(object):
    def __init__(self, rollout_steps):
        self.rollout_steps = rollout_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return self.rollout_steps

class RolloutGenerator(object):
    def __init__(self, env_func, num_actors, rollout_steps, rollout_queue, seed, device):
        self.env_func = env_func
        self.num_actors = num_actors
        self.rollout_steps = rollout_steps
        self.rollout_queue = rollout_queue
        self.seed = seed
        self.device = device
        self.env = None

    def run(self):
        self.env = VectorizedEnv([self.env_func(actor_id) for actor_id in range(self.num_actors)])
        states = self.env.reset()

        while True:
            try:
                rollout = Rollout(self.rollout_steps)

                for _ in range(self.rollout_steps):
                    actions = [self.env.action_space.sample() for _ in range(self.num_actors)]

                    rollout.states.append(states)

                    step_env_with_timeout = timeout(5)(lambda : self.env.step(actions))
                    states, rewards, dones, _ = step_env_with_timeout()

                    rollout.actions.append(actions)
                    rollout.rewards.append(rewards)
                    rollout.dones.append(dones)

                self.rollout_queue.put(rollout)

            except KeyboardInterrupt:
                self.env.close()
            except:
                self.env.close()
                for process in self.env.processes:
                    process.terminate()
                self.env = VectorizedEnv([self.env_func(actor_id) for actor_id in range(self.num_actors)])
                states = self.env.reset()