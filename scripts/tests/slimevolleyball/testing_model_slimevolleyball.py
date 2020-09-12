import slimevolleygym
import gym
import numpy as np
import torch

# States:
# 12-dim vector

# Actions:
# 3-dim vector
#
# action[0] > 0 -> move forward
# action[1] > 0 -> move backward
# action[2] > 0 -> jump
from nn.mlp import DeepMLPTorch


def random_policy():
    return np.random.choice([-1, 1], size=3)


def mlp_policy(env, model, input_size):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs = torch.from_numpy(obs).float()
        obs = obs[:input_size]
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)


def main():
    env = gym.make('SlimeVolley-v0')
    env.seed(13)
    input_size = 12
    output_size = 3
    hidden_sizes = [20, 128, 10, 5]
    model = DeepMLPTorch(input_size, output_size, *hidden_sizes)
    model.load_state_dict(
        torch.load(
            '../../../models/slimevolleyball/model-layers=12-[20, 128, 10, 5]-308-28-2020_17-30_NN=DeepMLPTorchIndividual_POPSIZE=40_GEN=20000_PMUTATION_0.2_PCROSSOVER_0.8_I=15040_SCORE=164.3799999999988.npy')
    )
    mlp_policy(env, model, input_size)


if __name__ == '__main__':
    main()
