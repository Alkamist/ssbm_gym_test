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
batch_size = 64
learn_every = 600
replay_buffer_size = 100000
num_actor_pools = 3
num_actors_per_pool = 4
rollout_steps = 300
seed = 1


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
    replay_buffer = ReplayBuffer(replay_buffer_size, device)

    total_rollouts = 0
    while True:
        try:
            rollout = melee_rollout_generator.generate_rollout()
        except:
            rollout = None

        if rollout is not None:
            total_rollouts += 1

            replay_buffer.add_rollout(rollout)

            print(melee_rollout_generator.rollout_queue.qsize())

#            if len(replay_buffer) > batch_size:
#                network.learn(replay_buffer.sample(batch_size), batch_size)
#
#            if total_rollouts % 60 == 0:
#                total_frames = total_rollouts * rollout_steps
#                print("Total Frames: %i" % total_frames)
#                network.save("checkpoints/agent" + str(total_frames) + ".pth")

    melee_rollout_generator.join_actor_pool_processes()