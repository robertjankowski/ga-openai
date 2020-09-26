import gym
import tensorflow as tf
import numpy as np

from scripts.multitask_learning.multitask_learning import MultiTaskModel, Shape

from gym import wrappers


def test_model(env_bipedalwalker, env_cartpole, model, bipedalwalker_shape, cartpole_shape):
    obs_bipedalwalker = env_bipedalwalker.reset()
    obs_cartpole = env_cartpole.reset()
    while True:
        env_bipedalwalker.render()
        env_cartpole.render()

        action = model({
            "first_input": np.reshape(obs_cartpole, (1, cartpole_shape.input)),
            # "first_input": np.reshape(np.random.rand(cartpole_shape.input), (1, cartpole_shape.input)),
            "second_input": np.reshape(obs_bipedalwalker, (1, bipedalwalker_shape.input))
        })
        action_cartpole, action_bipedalwalker = action
        action_cartpole = round(action_cartpole.numpy()[0].item())
        if action_cartpole > 1:
            action_cartpole = 1
        action_bipedalwalker = action_bipedalwalker.numpy()[0]

        obs_bipedalwalker, _, done_bipedalwalker, _ = env_bipedalwalker.step(action_bipedalwalker)
        obs_cartpole, _, done_cartpole, _ = env_cartpole.step(action_cartpole)

        if done_bipedalwalker:
            break


if __name__ == '__main__':
    env_bipedalwalker = gym.make('BipedalWalker-v2')
    env_cartpole = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, 'bipedalwalker_small_4_4_4', video_callable=lambda episode_id: True, force=True)

    env_bipedalwalker.seed(312)
    env_cartpole.seed(312)

    cartpole_shape = Shape(4, 1)
    bipedalwalker_shape = Shape(24, 4)
    hidden_sizes = [12, 20, 8]
    model = MultiTaskModel(cartpole_shape, bipedalwalker_shape, *hidden_sizes)
    model_path = tf.train.latest_checkpoint("../../multitask_learning/multitask-model-test")
    model.load_weights(model_path)

    test_model(env_bipedalwalker, env_cartpole, model.model, bipedalwalker_shape, cartpole_shape)

    env_bipedalwalker.close()
    env_cartpole.close()
