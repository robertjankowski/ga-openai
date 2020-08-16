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


def mlp_policy(env, model):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs = torch.from_numpy(obs).float()
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)


def main():
    env = gym.make('SlimeVolley-v0')

    model = DeepMLPTorch(12, 3, 20, 20)
    model.load_state_dict(
        torch.load(
            '../../../models/slimevolleyball/model-test08-16-2020_06-36_NN=DeepMLPTorchIndividual_POPSIZE=10_GEN=20_PMUTATION_0.1_PCROSSOVER_0.8_I=17_SCORE=65.09200000000001.npy'))
    mlp_policy(env, model)


if __name__ == '__main__':
    main()
