from melee_env import MeleeEnv

env = MeleeEnv(
    windows=True,
    render=True,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)
observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if reward != 0.0:
        print(env.action_space.data)

env.close()