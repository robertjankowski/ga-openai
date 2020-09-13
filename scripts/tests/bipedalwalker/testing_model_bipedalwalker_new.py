import gym
import torch

from nn.mlp import DeepMLPTorch


def test_mlp_torch(model, input_size: int, is_reduced=False):
    obs = env.reset()
    for _ in range(2000):
        env.render()
        obs = torch.from_numpy(obs).float()
        if is_reduced:
            obs = obs[:input_size]
        action = model.forward(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    # env = gym.wrappers.Monitor(env, 'bipedalwalker', video_callable=lambda episode_id: True, force=True)
    env.seed(123)

    INPUT_SIZE = 24
    HIDDEN_SIZE = [10, 12]
    OUTPUT_SIZE = 4

    model = DeepMLPTorch(INPUT_SIZE, OUTPUT_SIZE, *HIDDEN_SIZE)
    model.load(
        "../../../models/bipedalwalker/large_model/model-layers=24-[10, 12]-4-09-13-2020_13-23_NN=DeepBipedalWalkerIndividual_POPSIZE=10_GEN=10_PMUTATION_0.01_PCROSSOVER_0.8_I=10_SCORE=1.1157546929189408.npy"
    )
    test_mlp_torch(model, input_size=INPUT_SIZE, is_reduced=True)
    env.close()
