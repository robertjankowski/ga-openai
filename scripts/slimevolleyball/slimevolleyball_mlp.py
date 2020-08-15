from typing import Tuple

import copy
import gym
import slimevolleygym
import numpy as np
import random
import torch
import collections

from ga.individual import Individual, ranking_selection
from ga.population import Population
from nn.base_nn import NeuralNetwork
from nn.mlp import DeepMLPTorch

HIDDEN_SIZE = [10, 8]


def compare_list(l1, l2) -> bool:
    return collections.Counter(l1) == collections.Counter(l2)


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
        elapsed_episodes = 0
        alpha = 0.01
        for episode in range(n_episodes):
            if render:
                env.render()
            obs = torch.from_numpy(obs).float()
            action = self.nn.forward(obs)
            action = action.detach().numpy()
            obs, reward, done, _ = env.step(action)
            fitness += reward
            elapsed_episodes = episode
            if done:
                break
        return fitness + alpha * elapsed_episodes, self.nn.get_weights_biases()


def unroll_matrix(matrix: torch.Tensor, unroll_technic='vec') -> torch.Tensor:
    """
    unroll_technic: vec
    0 0 1 1
    0 0 1 1    ->  0 0 1 1 0 0 1 1 2 2 3 3 2 2 3 3
    2 2 3 3
    2 2 3 3

    unroll_technic: bvec (only for square matrix)
    0 0 1 1
    0 0 1 1    ->  0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
    2 2 3 3
    2 2 3 3

    :param matrix: weights matrix
    :param unroll_technic: {vec, bvec}
    :return: vector of weigths
    """
    if unroll_technic == 'vec':
        return matrix.flatten()
    else:
        pass


def get_random_layer(i1: DeepMLPTorchIndividual):
    return random.choice(list(i1.weights_biases.keys()))


def crossover(i1: DeepMLPTorchIndividual, i2: DeepMLPTorchIndividual) -> Tuple[DeepMLPTorchIndividual,
                                                                               DeepMLPTorchIndividual]:
    """
    Crossover
     1. select layer (weights matrix or vector of biases) to update
     2a. if weights matrix -> unroll matrix (two technics) -> and perform crossover at random position
     2b. if bias vector -> perform crossover at random position
     3. recreate initial matrix/vector
     4. return child1, child2

    """
    new_i1 = copy.deepcopy(i1)
    new_i2 = copy.deepcopy(i2)
    layer = get_random_layer(new_i1)
    layer_i1 = new_i1.weights_biases[layer]
    layer_i2 = new_i2.weights_biases[layer]
    layer_i1_size = list(layer_i1.size())
    layer_i2_size = list(layer_i2.size())
    if compare_list(layer_i1_size, layer_i2_size):
        # unroll weights matrix
        vec_i1 = unroll_matrix(layer_i1) if len(layer_i1_size) > 1 else layer_i1
        vec_i2 = unroll_matrix(layer_i2) if len(layer_i2_size) > 1 else layer_i2

        # choose randomly position
        size = list(vec_i1.size())[0]
        rand_position = random.randint(0, size - 1)
        child1 = vec_i1.detach().clone()
        child2 = vec_i2.detach().clone()

        # create children
        child_tmp = copy.deepcopy(child1)
        child1[rand_position:] = child2[rand_position:]
        child2[rand_position:] = child_tmp[rand_position:]

        # convert into initial shape
        child1 = child1.reshape(tuple(layer_i1_size))
        child2 = child2.reshape(tuple(layer_i2_size))

        # update layer
        new_i1.weights_biases[layer] = child1
        new_i2.weights_biases[layer] = child2

    return new_i1, new_i2


def mutation(i1: DeepMLPTorchIndividual, scale=10) -> DeepMLPTorchIndividual:
    new_i1 = copy.deepcopy(i1)
    layer = get_random_layer(new_i1)
    layer_i1 = i1.weights_biases[layer]
    layer_i1_size = tuple(layer_i1.size())
    # unroll weights matrix
    vec_i1 = unroll_matrix(layer_i1) if len(layer_i1_size) > 1 else layer_i1

    # choose N elements from uniform distribution X ~ U(-scale, scale)
    vec_i1_size = list(vec_i1.size())[0]
    n_elements = random.randint(1, vec_i1_size - 1)
    random_elements = torch.from_numpy(np.random.uniform(-scale, scale, n_elements))

    # randomly place new weights
    place = random.randint(0, vec_i1_size - n_elements - 1)
    vec_i1[place:(n_elements + place)] = random_elements
    vec_i1 = vec_i1.reshape(layer_i1_size)
    new_i1.weights_biases[layer] = vec_i1
    return i1


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
        if p_crossover > np.random.rand():
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        # Mutation
        if p_mutation > np.random.rand():
            child1 = mutation(child1)
            child2 = mutation(child2)

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)

        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


def main():
    env = gym.make('SlimeVolley-v0')
    env.seed(123)

    POPULATION_SIZE = 8
    MAX_GENERATION = 10
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
    p.set_population([DeepMLPTorchIndividual(INPUT_SIZE, 0, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)])
    p.run(env, generation, verbose=True, log=False)


if __name__ == '__main__':
    main()
