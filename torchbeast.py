import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing
import random
from copy import deepcopy

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from core import prof
from core import vtrace

from melee import Melee, melee_state_to_tensor, player_just_died

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
multiprocess_type = "spawn"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

disable_checkpoint = False
savedir = "checkpoints"
save_every = 1 # In minutes.

episode_steps = 3600
num_actors = 1
total_steps= 100000
batch_size = 8
unroll_length = 80
num_learner_threads = 2
use_lstm = True

# Loss settings.
entropy_cost = 0.0006
baseline_cost = 0.5
discounting = 0.99
reward_clipping = "abs_one" # Other option is "none".

# Optimizer settings.
learning_rate = 0.00048
alpha = 0.99
momentum = 0
epsilon = 0.01 # RMSProp epsilon
grad_norm_clipping = 40.0


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def calculate_reward(state, next_state):
    reward = 0.0

    if player_just_died(state, next_state, 1):
        reward = 1.0

    if player_just_died(state, next_state, 0):
        reward = -1.0

    return reward

def create_env(worker_id=0):
    return Environment(Melee(worker_id=worker_id, **options))

class Environment:
    def __init__(self, env):
        self.env = env
        self.state_size = env.state_size
        self.num_actions = env.num_actions
        self.episode_return = None
        self.episode_step = None
        self._state = None
        self._next_state = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        self._state = self.env.reset()
        initial_frame = melee_state_to_tensor(self._state).unsqueeze(0)
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        self._next_state = self.env.step(action.item())
        frame = melee_state_to_tensor(self._next_state).unsqueeze(0)
        reward = calculate_reward(self._state, self._next_state)
        done = self.episode_step[0].item() >= episode_steps

        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        self._state = deepcopy(self._next_state)

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self):
        self.env.close()


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


class Net(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Net, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions

        # FC output size + one-hot of last action + last reward.
        core_output_size = state_size + num_actions + 1
        self.core = nn.LSTM(input_size=core_output_size, hidden_size=core_output_size, num_layers=2)
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]
        T, B, _ = x.shape
        x = torch.flatten(x, 0, 1) # Merge time and batch.

        one_hot_last_action = F.one_hot(inputs["last_action"].view(T * B), self.num_actions).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        #seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        #gym_env.seed(seed)
        env = create_env(actor_index + 1)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def get_batch(
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state

def learn(
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats

def create_buffers(state_size, num_actions, num_buffers) -> Buffers:
    T = unroll_length
    specs = dict(
        frame=dict(size=(T + 1, state_size), dtype=torch.float32),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def train(): # pylint: disable=too-many-branches, too-many-statements
    xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    checkpointpath = "%s/%s%s" % (savedir, xpid, ".tar")

    num_buffers = max(2 * num_actors, batch_size)
    if num_actors >= num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if num_buffers < batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = unroll_length
    B = batch_size

    env = create_env()

    model = Net(env.state_size, env.num_actions)
    buffers = create_buffers(env.state_size, model.num_actions, num_buffers)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context(multiprocess_type)
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(env.state_size, env.num_actions).to(device=device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        eps=epsilon,
        alpha=alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, total_steps) / total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > save_every * 60:
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()

def test(num_episodes: int = 10):
    env = create_env()
    model = Net(env.state_size, env.num_actions)
    model.eval()
    checkpoint = torch.load("checkpoints/torchbeast-20200520-155918.tar", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    agent_state = model.initial_state(batch_size=1)
    observation = env.initial()
    returns = []

    while len(returns) < num_episodes:
        policy_outputs, agent_state = model(observation, agent_state)
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )


if __name__ == "__main__":
    train()
    #test(10)