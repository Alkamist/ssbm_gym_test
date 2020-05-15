import time

#from melee_env import MeleeEnv
from test_env2 import MeleeEnv
from DQN import Agent

options = dict(
    windows=True,
    render=False,
    speed=0,
    player1='ai',
    player2='cpu',
    char1='falcon',
    char2='falcon',
    stage='battlefield',
)

total_steps = 9999999999

if __name__ == "__main__":
    agent = Agent(state_size=792, action_size=30)
    #agent.load("checkpoints/agent.pth")

    env = MeleeEnv(frame_limit=total_steps, **options)
    observation = env.reset()

    start_time = time.time()
    start_step = 0

    for step_count in range(total_steps):
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(env.action_space.from_index(action))

        agent.step(observation, action, reward, next_observation, done)

        observation = next_observation

        # Every second print the FPS and Step Count.
        current_time = time.time()
        if current_time - start_time >= 1.0:
            start_time = current_time
            print("FPS:", step_count - start_step, "Step Count:", step_count)
            start_step = step_count

        # Every 15000 steps save the model.
        if step_count > 0 and step_count % 15000 == 0:
            agent.save("checkpoints/" + str(step_count) + ".pth")

    env.close()