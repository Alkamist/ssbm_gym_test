import threading

import torch

from melee_rollout_generator import MeleeRolloutGenerator
from replay_buffer import ReplayBuffer
from DQN import DQN


melee_options = dict(
    render=False,
    speed=0,
    player1='ai',
    player2='ai',
    char1='falcon',
    char2='falcon',
    stage='final_destination',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
replay_buffer_size = 250000
num_actor_pools = 3
num_actors_per_pool = 4
rollout_steps = 32
seed = 1


def gather_rollouts(rollout_generator, replay_buffer, learns_allowed, rollouts_generated):
    while True:
        rollout = melee_rollout_generator.generate_rollout()
        replay_buffer.add_rollout(rollout)
        learns_allowed[0] += 1
        rollouts_generated[0] += 1


if __name__ == "__main__":
    melee_rollout_generator = MeleeRolloutGenerator(
        num_actor_pools=num_actor_pools,
        num_actors_per_pool=num_actors_per_pool,
        rollout_steps=rollout_steps,
        seed=seed,
        device=device,
        dolphin_options=melee_options,
    )

    network = DQN(melee_rollout_generator.state_size, melee_rollout_generator.num_actions, device)
    network.load("checkpoints/agent.pth")

    replay_buffer = ReplayBuffer(replay_buffer_size, device)

    learns_allowed = [0]
    rollouts_generated = [0]

    gathering_thread = threading.Thread(target=gather_rollouts, args=(melee_rollout_generator, replay_buffer, learns_allowed, rollouts_generated))
    gathering_thread.start()

    need_to_save = False
    learn_iterations = 0
    while True:
        if learns_allowed[0] > 0 and len(replay_buffer) > batch_size:
            network.learn(replay_buffer.sample(batch_size))
            learn_iterations += 1
            learns_allowed[0] -= 1
            need_to_save = True

        if learn_iterations % 2000 == 0 and learn_iterations > 0 and need_to_save:
            print("Total Frames: {} / Learn Iterations: {}".format(rollouts_generated[0] * rollout_steps, learn_iterations))
            network.save("checkpoints/agent" + str(learn_iterations) + ".pth")
            need_to_save = False

    gathering_thread.join()
    melee_rollout_generator.join_actor_pool_processes()