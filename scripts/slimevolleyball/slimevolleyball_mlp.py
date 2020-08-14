from typing import Tuple

import gym
import slimevolleygym
import numpy as np
import random
import torch

from ga.individual import Individual, ranking_selection
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.mlp import DeepMLPTorch

HIDDEN_SIZE = [12, 2, 3]


class DeepMLPTorchIndividual(Individual):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.input_size = input_size

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return DeepMLPTorch(input_size, output_size, *HIDDEN_SIZE)

    def run_single(self, env, n_episodes=300, render=False) -> Tuple[float, np.array]:
        # Maybe reward should be combined with number of episodes??
        obs = env.reset()
        fitness = 0
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = torch.from_numpy(obs).float()
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            if done:
                break
        return fitness, self.nn.get_weights_biases()


def crossover(i1: DeepMLPTorchIndividual, i2: DeepMLPTorchIndividual) -> Tuple[DeepMLPTorchIndividual,
                                                                               DeepMLPTorchIndividual]:
    layer = random.choice(list(i1.weights_biases.keys()))

    # TODO:
    #  1. select layer (weights matrix or vector of biases) to update
    #  2a. if weights matrix -> unroll matrix (two technics) -> and perform crossover at random position
    #  2b. if bias vector -> perform crossover at random position
    #  3. recreate initial matrix/vector
    #  4. return child1, child2
    position = i1.weights_biases[layer]
    print(layer, position.size())


def mutation(i1: DeepMLPTorchIndividual) -> DeepMLPTorchIndividual:
    pass


def generation(env,
               old_population: list,
               new_population: list,
               p_mutation: float,
               p_crossover: float,
               p_inversion: float = 0.0):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1, parent2 = ranking_selection(old_population)

        # Crossover
        if p_crossover < np.random.rand():
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        # Mutation
        if p_mutation < np.random.rand():
            child1 = mutation(child1)

        # TODO
        # Check if either one of the child has better fitness score
        new_population[i] = old_population[i]
        new_population[i + 1] = old_population[i + 1]


def main():
    env = gym.make('SlimeVolley-v0')
    env.seed(123)

    POPULATION_SIZE = 6
    MAX_GENERATION = 3
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    INPUT_SIZE = 12
    OUTPUT_SIZE = 3

    assert POPULATION_SIZE % 2 == 0
    p = Population(DeepMLPTorchIndividual(INPUT_SIZE, 0, OUTPUT_SIZE),
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE,
                   CROSSOVER_RATE,
                   0.0)
    p.run(env, generation, verbose=True, log=False, output_folder='')


if __name__ == '__main__':
    main()
