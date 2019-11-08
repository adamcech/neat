import math
import random
import multiprocessing as mp
import numpy as np
import copy
import time


from typing import List, Tuple

from dataset.dataset import Dataset
from neat.species import Species
from neat.encoding.genotype import Genotype


class Population:
    __mutate_add_node_prop = 0.15
    __mutate_add_edge_prop = 0.3

    __mutate_weights_prop = 0.8

    __mutate_weights_perturbate = 0.9
    __mutate_weights_shift = 0.05

    __remove_percentage = 0.1

    __stagnation_time = 100

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
        
        self._population_elite = []
        self._species = []  # type: List[Species]
        self._stagnation_history = []

        initial_genotype = Genotype.initial_genotype(neat.dataset)

        self._population = [initial_genotype.initial_copy() for _ in range(self._size)]  # type: List[Genotype]

        self.evaluate(neat.dataset)

        self._grow_rate = 0.1
        self.max_species = 8  # int(self._size / 25) if self._size > 25 else -1
        self._compatibility = []

        self.elitism = 0.3

    def get_compatibility(self) -> float:
        return sum(self._compatibility) / len(self._compatibility)

    def get_grow_rate(self) -> float:
        return self._grow_rate

    def speciate(self):
        for species in self._species:
            species.reset()

        for genotype in self._population:
            species_found = False
            for species in self._species:
                is_compatible, compatibility = Genotype.is_compatible(self.neat, genotype, species.representative)
                if is_compatible:
                    species.add_member(genotype, compatibility)
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

        # Remove Extinct and Stagnating
        new_species = [species for species in self._species if not species.is_extinct()]

        # For not removing best 2 species
        species_fitness = np.argsort([s.fitness for s in new_species])[::-1]
        new_species = [new_species[species_fitness[i]] for i in range(len(species_fitness))]

        self._species = []
        for i in range(len(new_species)):
            species = new_species[i]
            if not species.is_stagnating() or i < 2:
                self._species.append(species)
                if species.is_stagnating():
                    species.score_history = []

        species_fitness = np.argsort([s.fitness for s in self._species])[::-1]

        # Limit max species
        if len(self._species) > self.max_species:
            self._species = [self._species[species_fitness[i]] for i in range(self.max_species)]

        # save compatibility
        self._compatibility = []
        for species in self._species:
            for genotype_compatibility in species.compatibility:
                self._compatibility.append(genotype_compatibility)

        # Population stagnation
        if len(self._stagnation_history) == self.__stagnation_time:
            if max(self._stagnation_history) == self._stagnation_history[0]:
                print("Population stagnation")
                if len(self._species) >= 2:
                    species_fitness = np.argsort([s.fitness for s in self._species])[::-1]
                    self._species = [self._species[species_fitness[0]], self._species[species_fitness[1]]]
                    self._stagnation_history = []

    """
    def crossover(self):
        self._species_champs = []
        new_pop = []

        species_fitness = [species.fitness for species in self._species]
        species_fitness_sum = sum(species_fitness)

        for species in self._species:
            slots = int(species.fitness / species_fitness_sum * self._size) - 1
            new_pop.append(species.get_champ())
            self._species_champs.append(species.get_champ())

            if species.is_dying():
                original_members = [member for member in species.members]
                for _ in range(slots):
                    original_genotype = random.sample(original_members, 1)[0]  # type: Genotype
                    diverse_genotype = Genotype.initial_copy(original_genotype)
                    new_size = len(diverse_genotype.edges) * (1 + ((1 - Genotype.compatibility_threshold) / 5))
                    while len(diverse_genotype.edges) < new_size:
                        if np.random.uniform() < 0.35:
                            diverse_genotype.mutate_add_node(self.neat.dataset)
                        else:
                            diverse_genotype.mutate_add_edge()
                    new_pop.append(diverse_genotype)
            else:
                for _ in range(slots):
                    # parents = random.sample(species.members, 2)
                    # new_pop.append(Genotype.crossover(parents[0], parents[1]))
                    parents = random.sample(species.members, 3)
                    new_pop.append(Genotype.triple_crossover(parents[0], parents[1], parents[2]))

        self._population = new_pop
    """

    def get_elitism_slots(self) -> int:
        return math.ceil(self._size * self.elitism)

    def append_elite(self, member: Genotype):
        if member not in self._population_elite:
            self._population.append(member)
            self._population_elite.append(member)

    def crossover(self):
        old_population = self._population
        old_population_best_indices = np.argsort([genotype.fitness for genotype in old_population])[::-1]
        species_fitness = [species.fitness for species in self._species]

        self._population = []
        self._population_elite = []

        for species in self._species:

            species_elite = species.get_elite(self.elitism)
            for genotype in species_elite:
                self.append_elite(genotype)

            slots = math.ceil(species.fitness / sum(species_fitness) * (self._size - self.get_elitism_slots()))

            is_dying = species.is_dying()
            for i in range(slots):
                if (np.random.uniform() < 0.97 and not is_dying) or len(self._species) == 1:
                    # Species mating
                    parents = random.sample(species.members, 3)
                else:
                    # Interspecies mating
                    parents = random.sample(species.members, 2)
                    other_species = random.sample([s for s in self._species if s != species], 1)[0]  # type: Species
                    parents.append(random.sample(other_species.members, 1)[0])

                child = Genotype.triple_crossover(parents[0], parents[1], parents[2])

                self._population.append(child)

        indices_counter = 0
        while len(self._population_elite) < self.get_elitism_slots():
            self.append_elite(old_population[old_population_best_indices[indices_counter]])
            indices_counter += 1

    def mutate_weights(self):
        # Perturbations for genomes not created by DE
        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            for edge in [edge for edge in genotype.edges if edge.mutable]:
                if np.random.uniform() < 0.8:
                    edge.mutate_perturbate_weight()
                if np.random.uniform() < 0.1:
                    edge.mutate_random_weight()

        # Smaller mutation for genomes created by DE
        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            for edge in [edge for edge in genotype.edges if not edge.mutable]:
                if np.random.uniform() < 0.1:
                    edge.mutate_random_weight()
                elif np.random.uniform() < 0.2:
                    edge.mutate_shift_weight()

        # Reenabling
        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            for edge in [edge for edge in genotype.edges if not edge.enabled]:
                if np.random.uniform() < 0.05:
                    edge.enabled = True

    """
    def mutate_weights(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_weights_prop and genotype not in self._species_champs:
                for edge in genotype.edges:
                    if edge.enabled:
                        if np.random.uniform(0, 1) < 0.9:
                            edge.mutate_perturbate_weight()
                        else:
                            edge.mutate_random_weight()
    """

    def mutate_topology(self):
        avg_compatibility = self.get_compatibility()
        threshold = Genotype.compatibility_threshold
        half_threshold = threshold + ((1 - threshold) * 0.5)
        diverse_threshold = threshold + ((1 - threshold) * 0.7)

        if len(self._species) + 2 < self.max_species or avg_compatibility > diverse_threshold:
            self._grow_rate += (avg_compatibility - threshold) / 2
        else:
            self._grow_rate *= 0.9

        if self._grow_rate > 1:
            self._grow_rate = 1
        if self._grow_rate < 0.1:
            self._grow_rate = 0.1

        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            if np.random.uniform() < self._grow_rate:
                new_size = len(genotype.edges) * 1.01
                while len(genotype.edges) < new_size:
                    if np.random.uniform() < 0.2:
                        genotype.mutate_add_node(self.neat.dataset)
                    if np.random.uniform() < 0.5:
                        genotype.mutate_add_edge()

    def mutate_add_node(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_node_prop and genotype not in self._population_elite:
                genotype.mutate_add_node(self.neat.dataset)

    def mutate_add_edge(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_edge_prop and genotype not in self._population_elite:
                genotype.mutate_add_edge()

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[i].score > self._population[best_fitness].score:
                best_fitness = i

        return self._population[best_fitness]

    def get_random(self) -> Genotype:
        return random.sample(self._population, 1)[0]

    def get_avg_score(self) -> float:
        return sum(p.score for p in self._population) / len(self._population)

    def get_avg_fitness(self) -> float:
        return sum(p.fitness for p in self._population) / len(self._population)

    def get_nodes_info(self) -> Tuple[float, int, int]:
        avg = 0.0
        maximum = 0
        minimum = 9999999

        for member in self._population:
            member_nodes = len(member.nodes)

            avg += member_nodes

            if member_nodes > maximum:
                maximum = member_nodes

            if member_nodes < minimum:
                minimum = member_nodes

        avg /= len(self._population)
        return avg, maximum, minimum

    def get_edges_info(self) -> Tuple[float, int, int]:
        avg = 0.0
        maximum = 0
        minimum = 9999999

        for member in self._population:
            member_edges = sum([1 for edge in member.edges if edge.enabled])

            avg += member_edges

            if member_edges > maximum:
                maximum = member_edges

            if member_edges < minimum:
                minimum = member_edges

        avg /= len(self._population)
        return avg, maximum, minimum

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

    """
    def evaluate(self, dataset: Dataset):
        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []

        step = int(len(self._population) / process_count)

        pid = 0
        start_time = time.time()

        # noinspection PyBroadException
        try:
            f = -step
            for i in range(process_count):
                pid += 1
                f += step
                t = f + step if i + 1 < process_count else len(self._population)
                process_out = mp.Queue()
                process_out_list.append(process_out)

                process_list.append(
                    mp.Process(target=self.eval_single,
                               args=(self._population[f:t], copy.deepcopy(dataset), process_out, pid, start_time)))

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
            print("Multi process evaluation failed, using single process!")
            for genotype in self._population:
                genotype.calculate_fitness(dataset)

        fitness = [genotype.fitness for genotype in self._population]

        m = min(fitness)
        m = abs(m) if m < 0 else 0

        self._stagnation_history.append(max(fitness))
        self._stagnation_history = self._stagnation_history[-self.__stagnation_time:]

        for genotype in self._population:
            genotype.score = genotype.fitness
            genotype.fitness += m

        print()
    """

    def evaluate(self, dataset: Dataset):
        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []
        pop_size = len(self._population)

        step = 4  # int(len(self._population) / process_count)

        # noinspection PyBroadException
        try:
            f = -step
            while f < pop_size:
                running = sum([1 for p in process_list if p.is_alive()])
                if running < process_count:
                    f += step
                    t = f + step
                    if t > pop_size:
                        t = pop_size

                    process_out = mp.Queue()
                    process_out_list.append(process_out)

                    p = mp.Process(target=self.eval_single,
                                   args=(self._population[f:t], copy.deepcopy(dataset), process_out))
                    p.start()
                    process_list.append(p)

                time.sleep(0.01)

            for p in process_list:
                p.join()

            # save results
            f = -step
            for i in range(len(process_out_list)):
                out_list = process_out_list[i]
                f += step
                t = f + step
                if t > pop_size:
                    t = pop_size

                for j in range(f, t):
                    self._population[j].fitness = out_list.get()
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self._population:
                genotype.calculate_fitness(dataset)

        fitness = [genotype.fitness for genotype in self._population]

        m = min(fitness)
        m = abs(m) if m < 0 else 0

        self._stagnation_history.append(max(fitness))
        self._stagnation_history = self._stagnation_history[-self.__stagnation_time:]

        for genotype in self._population:
            genotype.score = genotype.fitness
            genotype.fitness += m

    @staticmethod
    def eval_single(population: List[Genotype], dataset: Dataset, process_out: mp.Queue):
        for genotype in population:
            genotype.calculate_fitness(dataset)
            process_out.put(genotype.fitness)
