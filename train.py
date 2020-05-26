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

num_workers = 6


def run_melee(worker_id, out_queue):
    env = MeleeEnv(worker_id=worker_id, **melee_options)
    observation = env.reset()

    frames = 0
    while True:
        try:
            action = env.action_space.sample()

            step_env_with_timeout = timeout(5)(lambda : env.step(action))
            observation, reward, done, _ = step_env_with_timeout()

            frames += 1
            if frames % 3600 == 0:
                out_queue.put(frames)
                frames = 0
        except KeyboardInterrupt:
            env.close()
        except:
            out_queue.put(dict(worker_id=worker_id))

def start_melee_process(worker_id, out_queue):
    p = mp.Process(
        target=run_melee,
        args=(
            worker_id,
            out_queue
        )
    )
    p.start()
    return p


if __name__ == "__main__":
    test_queue = mp.Queue()

    workers = []
    for worker_id in range(num_workers):
        worker = start_melee_process(worker_id, test_queue)
        workers.append(worker)

    total_frames = 0
    while True:
        output = test_queue.get()

        if isinstance(output, dict):
            crashed_worker_id = output["worker_id"]
            workers[crashed_worker_id].terminate()
            workers[crashed_worker_id] = start_melee_process(crashed_worker_id, test_queue)
        else:
            total_frames += output
            if output > 0:
                print("Total Frames: %i" % total_frames)

    for p in workers:
        p.join()


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