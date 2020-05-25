import time
import queue

import torch
import torch.multiprocessing as mp

from melee_env import MeleeEnv


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

num_workers = 4


def run_melee(worker_id, out_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)
    observation = env.reset()

    frames = 0
    while True:
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        frames += 1
        if frames % 3600 == 0:
            out_queue.put(frames)
            frames = 0


if __name__ == "__main__":
    processes = []

    test_queue = mp.Queue()

    for worker_id in range(num_workers):
        p = mp.Process(
            target=run_melee,
            args=(
                worker_id,
                test_queue
            )
        )
        p.start()
        processes.append(p)

    total_frames = 0
    while True:
        frames = test_queue.get()
        total_frames += frames

        if frames > 0:
            print("Total Frames: %i" % total_frames)

    for p in processes:
        p.join()



#if __name__ == "__main__":
#    env = MeleeEnv(**melee_options)
#    observation = env.reset()
#
#    t = time.perf_counter()
#    i = 0
#    while True:
#        action = env.action_space.sample()
#        observation, reward, done, _ = env.step(action)
#
#        i += 1
#        t_ = time.perf_counter()
#        delta_t = t_ - t
#        if delta_t > 1.0:
#            print("FPS: %.1f" % i)
#            i = 0
#            t = t_