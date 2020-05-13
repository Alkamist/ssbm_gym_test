import numpy as np

from melee_env import MeleeEnv
from vectorized_env import VectorizedEnv

options = dict(
    windows=True,
    render=True,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

#env = MeleeEnv(**options)
#observation = env.reset()
#
#for _ in range(1000):
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#
#    if reward != 0.0:
#        print(reward)
#
#env.close()

#def get_env_maker(worker_id=0):
#    def make_env():
#        return MeleeEnv(worker_id=worker_id, **options)
#    return make_env
#
#if __name__ == "__main__":
#    env = VectorizedEnv([get_env_maker(i) for i in range(2)])
#    observation = env.reset()
#
#    for _ in range(1000):
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step([action for _ in range(2)])
#
#        #if reward != 0.0:
#        #    print(env.action_space.data)
#
#    env.close()