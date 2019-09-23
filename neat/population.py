import random

import numpy as np
from typing import List

from dataset.dataset import Dataset
from neat.encoding.genotype import Genotype
from neat.encoding.mutation_edge import MutationEdge
from neat.encoding.mutation_node import MutationNode


class Population:
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
        self._population = [initial_genotype.get_population_copy() for _ in range(self._size)]  # type: List[Genotype]
        self.evaluate(dataset)

        self._species = []  # type: List[List[Genotype]]
        self._first_speciation()

    def evaluate(self, dataset: Dataset):
        for genotype in self._population:
            genotype.evaluate(dataset)

    def _evaluate_shared_fitness(self):
        for s in self._species:
            for genotype in s:
                genotype.evaluate_shared_fitness(self._c1, self._c2, self._c3, self._t, s)

    def _remove_worst(self, ind_fitness: List[List[float]]):
        for i in range(len(self._species)):
            indices = np.argsort(ind_fitness[i])
            best_ind = [self._species[i][indices[-j-1]] for j in range(min(len(indices), int(self._size * 0.25)))]
            self._species[i] = best_ind

    def crossover(self):
        new_population = []  # type: List[Genotype]

        """
        # Survivors
        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.25:
                new_population.append(genotype)
        """

        # The Best survivors
        fitness = [genotype.get_fitness() for genotype in self._population]
        fitness_sort = np.argsort(np.array(fitness))

        for i in range(0, int(self._size * 0.25)):
            if np.random.uniform(0, 1) < 0.25:
                new_population.append(self._population[fitness_sort[-i-1]])

        self._speciate()
        self._evaluate_shared_fitness()
        species_fitness = [sum([ind.get_shared_fitness() for ind in species]) for species in self._species]
        self._remove_worst([[genotype.get_fitness() for genotype in species] for species in self._species])
        ind_fitness = [[ind.get_fitness() for ind in s] for s in self._species]

        while len(new_population) < self._size:
            selected_species = self._get_proportional_select(species_fitness)

            if (len(self._species) == 1 or np.random.uniform(0, 1) >= 0.001) and len(self._species[selected_species]) >= 2:
                mom = self._species[selected_species][self._get_proportional_select(ind_fitness[selected_species])]

                dad = mom
                while dad == mom:
                    dad = self._species[selected_species][self._get_proportional_select(ind_fitness[selected_species])]
            else:
                other_species = selected_species
                while other_species == selected_species:
                    other_species = self._get_proportional_select(species_fitness)

                mom = self._species[selected_species][self._get_proportional_select(ind_fitness[selected_species])]
                dad = self._species[other_species][self._get_proportional_select(ind_fitness[other_species])]

            new_population.append(Genotype.crossover(mom, dad))

        self._population = new_population

    @staticmethod
    def _get_proportional_select(species_fitness: List[float]) -> int:
        species_random = np.random.uniform(0, sum(species_fitness))

        for i in range(len(species_fitness)):
            if species_random < species_fitness[i]:
                return i
            else:
                species_random -= species_fitness[i]

    def _speciate(self):
        # Creating representatives
        new_species = [[s[0]] for s in self._species]  # type: List[List[Genotype]]

        for genotype in self._population:
            found_species = False
            species_compatibility = {}

            for i in range(len(self._species)):
                species_genotype = self._species[i][0]
                compatibility = Genotype.calculate_compatibility(self._c1, self._c2, self._c3, genotype, species_genotype)

                if compatibility < self._t:
                    species_compatibility[i] = compatibility
                    found_species = True

            if found_species:
                selected_species = min(species_compatibility, key=species_compatibility.get)
                if genotype not in new_species[selected_species]:
                    new_species[selected_species].append(genotype)
            else:
                new_species.append([genotype])

        self._species = new_species

    def _first_speciation(self):
        for genotype in self._population:
            found_species = False
            species_compatibility = {}

            for i in range(len(self._species)):
                species_genotype = self._species[i][0]
                compatibility = Genotype.calculate_compatibility(self._c1, self._c2, self._c3, genotype, species_genotype)

                if compatibility < self._t:
                    species_compatibility[i] = compatibility
                    found_species = True

            if found_species:
                self._species[min(species_compatibility, key=species_compatibility.get)].append(genotype)
            else:
                self._species.append([genotype])

    def mutate_weights(self):
        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.8:
                for edge in genotype.edges:
                    if np.random.uniform(0, 1) < 0.9:
                        edge.mutate_perturbate_weight()
                    else:
                        edge.mutate_random_weight()

    def mutate_add_node(self):
        mutations = []  # type: List[MutationNode]

        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.03:
                genotype.mutate_add_node(mutations)

    def mutate_add_edge(self):
        mutations = []  # type: List[MutationEdge]

        for genotype in self._population:
            if np.random.uniform(0, 1) < 0.05:
                genotype.mutate_add_edge(mutations)

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[i].get_fitness() > self._population[best_fitness].get_fitness():
                best_fitness = i

        return self._population[best_fitness]

    def print_all_fitness(self):
        ret = ""

        for genotype in self._population:
            ret += str(genotype.get_fitness()) + ", "

        print(ret)
