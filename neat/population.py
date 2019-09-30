import random

import numpy as np
from typing import List

from dataset.dataset import Dataset
from neat.species import Species
from neat.encoding.genotype import Genotype


class Population:
    __mutate_add_node_prop = 0.03
    __mutate_add_edge_prop = 0.05
    __mutate_weights_prop = 0.8
    __interspecies_mating_prop = 0.001
    __remove_percentage = 0.1

    """Population representation

    Args:
        c1 (float): Excess Genes Importance
        c2 (float): Disjoint Genes Importance
        c3 (float): Weights difference Importance
        t (float): Compatibility Threshold
        size (int): Population size
        dataset (Dataset): Dataset to use
    """

    def __init__(self, c1: float, c2: float, c3: float, t: float, size: int, dataset: Dataset):
        self._size = size

        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
        self._t = t

        initial_genotype = Genotype.initial_genotype(dataset)
        self._population = [initial_genotype.initial_copy() for _ in range(self._size)]  # type: List[Genotype]

        self._species = []  # type: List[Species]

        self.evaluate(dataset)

    def speciate(self):
        for species in self._species:
            species.reset()

        for genotype in self._population:
            species_found = False
            for species in self._species:
                if self._t > Genotype.calculate_compatibility(self._c1, self._c2, self._c3, genotype,
                                                              species.representative):
                    species.add_member(genotype)
                    species_found = True
                    break

            if not species_found:
                self._species.append(Species(genotype))

        # Has to be calculated with ALL members before removing worst
        for species in self._species:
            species.evaluate_fitness()

        # Removing Worst
        for species in self._species:
            species.remove_worst(Population.__remove_percentage)

        # Remove Extinct
        self._species = [species for species in self._species if len(species.members) > 1]

    def crossover(self):
        new_pop = []

        species_fitness = [species.get_fitness() for species in self._species]
        species_fitness_sum = sum(species_fitness)

        for species in self._species:
            slots = int(species.get_fitness() / species_fitness_sum * self._size) - 1
            new_pop.append(species.get_champ())

            for _ in range(slots):
                if len(self._species) == 1 or np.random.uniform(0, 1) > 0.001:
                    parents = random.sample(species.members, 2)
                    new_pop.append(Genotype.crossover(parents[0], parents[1]))
                else:
                    other_species = species
                    while other_species == species:
                        other_species = random.sample(self._species, 1)[0]

                    mom = random.sample(species.members, 1)[0]
                    dad = random.sample(other_species.members, 1)[0]
                    new_pop.append(Genotype.crossover(mom, dad))

        self._population = new_pop

    def mutate_weights(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_weights_prop:
                for edge in genotype.edges:
                    if edge.enabled:
                        if np.random.uniform(0, 1) < 0.9:
                            edge.mutate_perturbate_weight()
                        else:
                            edge.mutate_random_weight()

    def mutate_add_node(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < self.__mutate_add_node_prop:
                genotype.mutate_add_node()

    def mutate_add_edge(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < self.__mutate_add_edge_prop:
                genotype.mutate_add_edge()

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[i].get_fitness() > self._population[best_fitness].get_fitness():
                best_fitness = i

        return self._population[best_fitness]

    def get_species(self) -> List[Species]:
        return self._species

    @staticmethod
    def _get_natural_select(fitness: List[float]) -> int:
        rand = np.random.uniform(0, sum(fitness))

        for i in range(len(fitness)):
            if rand < fitness[i]:
                return i
            else:
                rand -= fitness[i]

    def evaluate(self, dataset: Dataset):
        for genotype in self._population:
            genotype.calculate_fitness(dataset)
