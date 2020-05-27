import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv
from rollout_generator import RolloutGenerator
from experience_buffer import ExperienceBuffer
from learner import Learner
from models import Policy


melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
    act_every=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_actors = 2
batch_size = 64
rollout_steps = 600
seed = 1


def get_melee_env_func(actor_id):
    return lambda : MeleeEnv(worker_id=actor_id, **melee_options)


if __name__ == "__main__":
    experience_buffer = ExperienceBuffer(batch_size=batch_size)

    shared_state_dict = Policy(MeleeEnv.observation_size, MeleeEnv.num_actions)
    shared_state_dict.share_memory()

    rollout_queue = mp.Queue()

    rollout_generator = RolloutGenerator(
        env_func = get_melee_env_func,
        num_actors = num_actors,
        rollout_steps = rollout_steps,
        rollout_queue = rollout_queue,
        shared_state_dict = shared_state_dict,
        seed = seed,
        device = device
    )

    learner = Learner(
        observation_size=MeleeEnv.observation_size,
        num_actions=MeleeEnv.num_actions,
        lr=3e-5,
        discounting=0.99,
        baseline_cost=0.5,
        entropy_cost=0.0025,
        grad_norm_clipping=40.0,
        save_interval=2,
        seed=seed,
        shared_state_dict=shared_state_dict,
        device=device,
    )

    process = mp.Process(target=rollout_generator.run)
    process.start()

    total_steps = 0
    num_traces = 0
    while True:
        rollout = rollout_queue.get()

        total_steps += len(rollout) * num_actors
        #print(total_steps)

        experience_buffer.add(rollout.states, rollout.actions, rollout.rewards, rollout.dones, rollout.logits)

        if experience_buffer.batch_is_ready:
            states_batch, actions_batch, rewards_batch, dones_batch, logits_batch = experience_buffer.get_batch()
            learner.learn(states_batch, actions_batch, rewards_batch, dones_batch, logits_batch)

    process.join()