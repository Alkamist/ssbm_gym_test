from follow_env import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=True,
    speed=1,
    player1='ai',
    player2='human',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 100000

if __name__ == "__main__":
    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    agent = Agent(gamma=0.99, epsilon=0.0, batch_size=64, lr=0.001, n_actions=env.action_space.n, input_dims=env.observation_space.n)
    agent.load("checkpoints/agent.pth")
    agent.evaluate()

    for step_count in range(total_steps):
        action = agent.choose_action(observation)
        _, reward, done, info = env.step(env.action_space.from_index(action))

    env.close()