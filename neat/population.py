import random

import numpy as np
from typing import List

from dataset.dataset import Dataset
from neat.encoding.genotype import Genotype


class Population:
    """Population representation
    """

    def __init__(self, size: int, dataset: Dataset):
        self._size = size

        initial_genotype = Genotype(dataset)
        self._population = [initial_genotype.get_population_copy() for _ in range(self._size)]  # type: List[Genotype]

        self.evaluate(dataset)

    def evaluate(self, dataset: Dataset):
        for genotype in self._population:
            genotype.evaluate(dataset)

    def crossover(self):
        # Crossover
        if np.random.uniform(0, 1) < 0.75:
            if np.random.uniform(0, 1) < 0.001:
                # Interspecies mating
                pass
            else:
                # Species mating
                pass

    def mutate_weights(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.8:
                for edge in genotype.edges:
                    if np.random.uniform(0.1) < 0.9:
                        edge.mutate_perturbate_weight()
                        pass
                    else:
                        edge.mutate_random_weight()

    def mutate_add_node(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.03:
                genotype.mutate_add_node(random.sample(genotype.edges, 1)[0])

    def mutate_add_edge(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.05:
                genotype.mutate_add_edge()

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[i].get_fitness() > self._population[best_fitness].get_fitness():
                best_fitness = i

        return self._population[best_fitness]
