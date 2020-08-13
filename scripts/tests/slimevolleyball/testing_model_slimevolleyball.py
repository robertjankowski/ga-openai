import slimevolleygym
import gym
import numpy as np


# States:
# 12-dim vector

# Actions:
# 3-dim vector
#
# action[0] > 0 -> move forward
# action[1] > 0 -> move backward
# action[2] > 0 -> jump

def random_policy():
    return np.random.choice([-1, 1], size=3)


def main():
    env = gym.make('SlimeVolley-v0')

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = random_policy()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(total_reward)


if __name__ == '__main__':
    main()
