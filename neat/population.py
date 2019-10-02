import random
import multiprocessing as mp
import numpy as np
import copy

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

    def __init__(self, neat: "Neat"):
        self.neat = neat
        self._size = neat.population_size

        self._champs = []
        self._species = []  # type: List[Species]

        initial_genotype = Genotype.initial_genotype(neat.dataset)
        self._population = [initial_genotype.initial_copy() for _ in range(self._size)]  # type: List[Genotype]

        self.evaluate(neat.dataset)

    def speciate(self):
        for species in self._species:
            species.reset()

        for genotype in self._population:
            species_found = False
            for species in self._species:
                if Genotype.is_compatible(self.neat, genotype, species.representative):
                    species.add_member(genotype)
                    species_found = True
                    break

            if not species_found:
                self._species.append(Species(genotype))

        # calculated with ALL members before removing worst
        for species in self._species:
            species.evaluate_fitness()

        # Removing Worst
        for species in self._species:
            species.remove_worst(Population.__remove_percentage)

        # Remove Extinct
        self._species = [species for species in self._species if len(species.members) > 1]

    def crossover(self):
        self._champs = []
        new_pop = []

        species_fitness = [species.get_fitness() for species in self._species]
        species_fitness_sum = sum(species_fitness)

        for species in self._species:
            slots = int(species.get_fitness() / species_fitness_sum * self._size) - 1
            new_pop.append(species.get_champ())
            self._champs.append(species.get_champ())

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
            if np.random.uniform() < self.__mutate_weights_prop and genotype not in self._champs:
                for edge in genotype.edges:
                    if edge.enabled:
                        if np.random.uniform(0, 1) < 0.9:
                            edge.mutate_perturbate_weight()
                        else:
                            edge.mutate_random_weight()

    def mutate_add_node(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_node_prop and genotype not in self._champs:
                genotype.mutate_add_node()

    def mutate_add_edge(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_edge_prop and genotype not in self._champs:
                genotype.mutate_add_edge()

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[i].fitness > self._population[best_fitness].fitness:
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
        process_count = 3
        process_list = []  # type: List[mp.Process]
        process_out_list = []

        step = int(len(self._population) / process_count)

        # noinspection PyBroadException
        try:
            f = -step
            for i in range(process_count):
                f += step
                t = f + step if i + 1 < process_count else len(self._population)
                process_out = mp.Queue()
                process_out_list.append(process_out)

                process_list.append(
                    mp.Process(target=self.eval_single,
                               args=(self._population[f:t], copy.deepcopy(dataset), process_out)))

            for p in process_list:
                p.start()

            for p in process_list:
                p.join()

            f = -step
            for i in range(process_count):
                out_list = process_out_list[i]
                f += step
                t = f + step if i + 1 < process_count else len(self._population)

                for j in range(f, t):
                    self._population[j].fitness = out_list.get()
        except Exception:
            print("Multi process evaluation failed")
            for genotype in self._population:
                genotype.calculate_fitness(dataset)

    @staticmethod
    def eval_single(population: List[Genotype], dataset: Dataset, process_out: mp.Queue):
        for genotype in population:
            genotype.calculate_fitness(dataset)
            print("Fitness: " + str(genotype.fitness))
            process_out.put(genotype.fitness)
