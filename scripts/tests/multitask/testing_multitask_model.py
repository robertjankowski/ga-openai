import gym
import tensorflow as tf
from scripts.multitask_learning.multitask_learning import MultiTaskModel, Shape

from gym import wrappers


# TODO
# 1. model should have two inputs - play games simultaneously?
# 2. output also should be converted to right format


def test_mlp_torch(model, input_size: int, is_reduced=False):
    obs = env.reset()
    for _ in range(2000):
        env.render()
        if is_reduced:
            obs = obs[:input_size]
        action = model(obs)
        action = action.detach().numpy()
        obs, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    # env = gym.wrappers.Monitor(env, 'bipedalwalker_small_4_4_4', video_callable=lambda episode_id: True, force=True)
    env.seed(312)

    cartpole_shape = Shape(4, 1)
    bipedalwalker_shape = Shape(24, 4)
    hidden_sizes = [12, 20, 8]
    model = MultiTaskModel(cartpole_shape, bipedalwalker_shape, *hidden_sizes)
    model_path = tf.train.latest_checkpoint("../../multitask_learning/multitask-model-test")
    model.load_weights(model_path)

    test_mlp_torch(model.model, input_size=bipedalwalker_shape.input, is_reduced=True)

    env.close()
