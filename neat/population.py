import math
import random

import numpy as np


from typing import List, Tuple

from dataset.dataset import Dataset
from neat.config import Config
from neat.encoding.genotype_trainer import GenotypeTrainer
from neat.encoding.innovation_map import InnovationMap
from neat.evaluator import Evaluator
from neat.species import Species
from neat.encoding.genotype import Genotype


class Population:
    __mutate_add_node_prop = 0.15
    __mutate_add_edge_prop = 0.3

    __mutate_weights_prop = 0.8

    __mutate_weights_perturbate = 0.9
    __mutate_weights_shift = 0.05

    __remove_percentage = 0.0

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

    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        self.innovation_map = InnovationMap()

        self._size = config.population_size
        
        self._population_elite = []
        self._species = []  # type: List[Species]
        self._stagnation_history = []

        initial_genotype = Genotype.initial_genotype(self.dataset, self.innovation_map)

        self._population = [initial_genotype.initial_copy() for _ in range(self._size)]  # type: List[Genotype]
        self.seed = [328300]
        Evaluator(self._population, self.dataset, True, self.seed).calculate_score()

        self._offspring_population = []  # type: List[Genotype]
        self._offspring_mutable = []  # type: List[bool]

        self._topology_grow_rate = 0.1
        self.max_species = 4  # int(self._size / 25) if self._size > 25 else -1
        self._compatibility = []

        self.species_elitism = 0.2
        self.species_elite = []  # type: List[Genotype]

    def remove_recursions(self):
        for genotype in self._population:
            genotype.ancestor = None
            genotype.species = None
        self._offspring_population = []
        self.species_elite = []
        self._population_elite = []

    def next_seed(self, count):
        second_best = self._population[np.argsort([g.score for g in self._population])[-2]]
        self.seed = Evaluator([second_best], self.dataset, True, random.sample(range(10000000), 160)).find_best_seed(count)

    def adjust_fitness(self):
        full_pop = self._population + self._offspring_population

        m = min([genotype.score for genotype in full_pop])
        m = abs(m) if m < 0 else 0

        if m != 0:
            for genotype in full_pop:
                genotype.fitness = genotype.score + m

    def speciate(self):
        self.adjust_fitness()

        for species in self._species:
            species.reset()

        for genotype in self._population:
            species_found = False
            best_species = -1
            best_compatibility = 0
            for i in range(len(self._species)):
                is_compatible, compatibility = self._species[i].calculate_compatibility(genotype)
                if is_compatible:
                    species_found = True
                    if compatibility > best_compatibility:
                        best_compatibility = compatibility
                        best_species = i

            if species_found:
                genotype.species = self._species[best_species]
                self._species[best_species].add_member(genotype, best_compatibility)
            else:
                new_species = Species(genotype)
                genotype.species = new_species
                self._species.append(new_species)

        self._species = [species for species in self._species if not species.is_empty()]

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

        # Limit max species
        if len(self._species) > self.max_species:
            species_fitness = np.argsort([s.fitness for s in self._species])[::-1]
            best_member_species = self._species[np.argsort([s.get_best_member().score for s in self._species])[-1]]
            new_species = [best_member_species]

            species_counter = 0
            while len(new_species) < self.max_species:
                if self._species[species_fitness[species_counter]] not in new_species:
                    new_species.append(self._species[species_fitness[species_counter]])
                species_counter += 1

            self._species = new_species

        # save compatibility
        self._compatibility = []
        for species in self._species:
            for genotype_compatibility in species.compatibility:
                self._compatibility.append(genotype_compatibility)

        self.species_elite = []
        elite = [species.get_elite(self.species_elitism) for species in self._species]
        for e in elite:
            self.species_elite.extend(e)

    def crossover(self):
        self._classic_crossover()
        # self._stagnation_crossover() if self._is_stagnating() else self._classic_crossover()

    def train_nets(self):
        self._species = []
        self._compatibility = [1]
        self._topology_grow_rate = 0.1
        self._stagnation_history = []

        genotype_trainer = GenotypeTrainer(self.dataset, self._population, self.config)

        genotype_trainer.train()
        self._population, self.seed = genotype_trainer.merge()
        self.new_evaluate_ancestors()

    def _stagnation_crossover(self):
        print("Population stagnation")

        self._stagnation_history = []
        self._population = []
        self._population_elite = []
        self._topology_grow_rate = 0.1

        species_fitness = [species.fitness for species in self._species]

        for species in self._species:
            species_best = species.get_best_member()
            slots = math.ceil(species.fitness / sum(species_fitness) * self._size)
            for _ in range(slots):
                self._population.append(species_best.initial_copy())

    def _classic_crossover(self):
        species_fitness = [species.fitness for species in self._species]

        self._population = []
        self._population_elite = []

        for species in self._species:
            species_elite = species.get_elite(self.species_elitism)
            for genotype in species_elite:
                self.append_elite(genotype)

            slots = math.ceil(species.fitness / sum(species_fitness) * self._size) - len(species_elite)

            is_dying = species.is_dying()
            for i in range(slots):
                if (np.random.uniform() < 0.99 and not is_dying) or len(self._species) == 1:
                    # Species mating
                    parents = random.sample(species.members, 3)
                else:
                    # Interspecies mating
                    parents = random.sample(species.members, 2)
                    other_species = random.sample([s for s in self._species if s != species], 1)[0]  # type: Species
                    parents.append(random.sample(other_species.members, 1)[0])

                # Best member evolution
                if i == 0 and species.get_best_member() not in parents:
                    parents[0] = species.get_best_member()

                child = Genotype.triple_crossover(parents[0], parents[1], parents[2])
                self._population.append(child)

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
                if np.random.uniform() < 0.03:
                    edge.mutate_random_weight()
                elif np.random.uniform() < 0.2:
                    edge.mutate_shift_weight()

        # Reenabling
        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            for edge in [edge for edge in genotype.edges if not edge.enabled]:
                if np.random.uniform() < 0.01:
                    edge.enabled = True

    def mutate_topology(self):
        avg_compatibility = self.get_compatibility()
        threshold = Genotype.compatibility_threshold
        diverse_threshold = threshold + ((1 - threshold) * 0.7)

        if len(self._species) < self.max_species or avg_compatibility > diverse_threshold:
            self._topology_grow_rate += (avg_compatibility - threshold)
        else:
            self._topology_grow_rate *= 0.9

        if self._topology_grow_rate > 1:
            self._topology_grow_rate = 1
        if self._topology_grow_rate < 0.1:
            self._topology_grow_rate = 0.1

        for genotype in [genotype for genotype in self._population if genotype not in self._population_elite]:
            if np.random.uniform() < self._topology_grow_rate:
                new_size = len(genotype.edges) * 1.02
                while len(genotype.edges) < new_size:
                    if np.random.uniform() < 0.1:
                        genotype.mutate_add_node(self.dataset, self.innovation_map)
                    else:
                        genotype.mutate_add_edge(self.dataset, self.innovation_map)

    def mutate_add_node(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_node_prop and genotype not in self._population_elite:
                genotype.mutate_add_node(self.dataset, self.innovation_map)

    def mutate_add_edge(self):
        for genotype in self._population:
            if np.random.uniform() < self.__mutate_add_edge_prop and genotype not in self._population_elite:
                genotype.mutate_add_edge(self.dataset, self.innovation_map)

    def is_stagnating(self):
        return max(self._stagnation_history) == self._stagnation_history[0] and len(self._stagnation_history) == self.__stagnation_time

    def get_species(self) -> List[Species]:
        return self._species

    def get_grow_rate(self) -> float:
        return self._topology_grow_rate

    def get_compatibility(self) -> float:
        return sum(self._compatibility) / len(self._compatibility)

    def get_avg_score(self) -> float:
        return sum(p.score for p in self._population) / len(self._population)

    def get_avg_fitness(self) -> float:
        self.adjust_fitness()
        return sum(p.fitness for p in self._population) / len(self._population)

    def get_random_member(self) -> Genotype:
        return random.sample(self._population, 1)[0]

    def get_best_member(self) -> Genotype:
        best_score = 0
        for i in range(1, len(self._population)):
            if self._population[i].score > self._population[best_score].score:
                best_score = i

        return self._population[best_score]

    def get_worst_member(self) -> Genotype:
        worst_score = 0
        for i in range(1, len(self._population)):
            if self._population[i].score < self._population[worst_score].score:
                worst_score = i

        return self._population[worst_score]

    def append_elite(self, member: Genotype):
        if member not in self._population_elite:
            self._population.append(member)
            self._population_elite.append(member)

    def get_nodes_info(self) -> Tuple[float, int, int]:
        inputs = self.config.bias_size + self.config.input_size

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
        return avg - inputs, maximum - inputs, minimum - inputs

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

    def evaluate(self):
        Evaluator(self._population, self.dataset, False, random.sample(range(1000000), 1)).calculate_score()

    def save_stats(self):
        if len(self._species) + 1 >= self.max_species:
            self._stagnation_history.append(max([genotype.score for genotype in self._population]))
            self._stagnation_history = self._stagnation_history[-self.__stagnation_time:]
        else:
            self._stagnation_history = []

    @staticmethod
    def __get_natural_select(fitness: List[float]) -> int:
        rand = np.random.uniform(0, sum(fitness))

        for i in range(len(fitness)):
            if rand < fitness[i]:
                return i
            else:
                rand -= fitness[i]

    def new_crossover(self):
        self._offspring_population = []
        self._offspring_mutable = []
        species_slots = self._get_species_slots()
        is_low_compatibility = self.get_compatibility() < (Genotype.compatibility_threshold + ((1 - Genotype.compatibility_threshold) * 0.75))

        for i in range(len(self._species)):
            species = self._species[i]
            slots = species_slots[i]
            species_elite = species.get_elite(self.species_elitism)

            for j in range(slots):
                ancestor = species.members[j % len(species.members)]

                mutable = (len(self._species) == self.max_species and is_low_compatibility) or np.random.uniform() < 0.5
                self._offspring_mutable.append(mutable)

                if mutable:
                    if np.random.uniform() < 0.97 or len(self._species) == 1:
                        # Species mating
                        parents = random.sample(species.members, 2)
                    else:
                        # Interspecies mating
                        parents = random.sample(self._population, 2)

                    best = random.sample(species_elite, 1)[0]
                    mom = parents[0]
                    dad = parents[1]

                    offspring = Genotype.new_triple_crossover(ancestor, best, mom, dad)
                    self._offspring_population.append(offspring)
                else:
                    offspring = Genotype.new_mutable_copy(ancestor)

                    if np.random.uniform() < 0.1:
                        offspring.mutate_add_node(self.dataset, self.innovation_map)
                    else:
                        offspring.mutate_add_edge(self.dataset, self.innovation_map)

                    self._offspring_population.append(offspring)

    def new_boost_species(self):
        if any(species.is_dying() for species in self._species):
            species_slots = self._get_species_slots()
            slots_diff = [species_slots[i] - len(self._species[i].members) for i in range(len(self._species))]

            casualties = [genotype for genotype in self._population if genotype.species is None or genotype.species not in self._species]  # type: List[Genotype]

            for i in range(len(self._species)):
                if slots_diff[i] < 0:
                    casualties.extend(self._species[i].get_casualties(abs(slots_diff[i])))

            for i in range(len(self._species)):
                species = self._species[i]
                species_slots_diff = slots_diff[i]

                if species.is_dying() and species_slots_diff > 0:
                    casualties_indices = np.argsort([species.calculate_compatibility(c)[1] for c in casualties])[::-1]
                    selected_casualties = [casualties[casualties_indices[j]] for j in range(min(len(casualties), species_slots_diff))]  # type: List[Genotype]

                    casualties = [c for c in casualties if c not in selected_casualties]
                    species.add_casualties(selected_casualties)

    def _get_species_slots(self) -> List[int]:
        if len(self._species) == 1:
            return [self._size]

        """
        species_fitness = [species.fitness for species in self._species]

        curr_slots = [len(species.members) for species in self._species]
        species_share = [species.fitness / sum(species_fitness) for species in self._species]
        default_slots = [math.floor(species_share[i] * self._size) for i in range(len(self._species))]
        diff_slots = [default_slots[i] - curr_slots[i] for i in range(len(self._species))]

        added_slots = [curr_slots[i] + 10 if diff_slots[i] > 10 else default_slots[i] for i in range(len(self._species))]
        empty_slots = self._size - sum(added_slots)

        shares_ratio = sum([species_share[i] for i in range(len(species_share)) if diff_slots[i] <= 10])
        species_new_share = [0 if diff_slots[i] > 10 else species_share[i] / shares_ratio for i in range(len(self._species))]
        new_slots = [added_slots[i] if diff_slots[i] > 10 else default_slots[i] + math.floor(species_new_share[i] * empty_slots) for i in range(len(self._species))]

        return new_slots
        """

        species_fitness = [species.fitness for species in self._species]
        new_slots = [math.floor(species.fitness / sum(species_fitness) * self._size) for species in self._species]
        return new_slots

    def new_mutations(self):
        for genotype in [self._offspring_population[i] for i in range(len(self._offspring_population)) if self._offspring_mutable[i]]:

            # Topology mutations
            if np.random.uniform() < 0.05:
                genotype.mutate_add_node(self.dataset, self.innovation_map)
            if np.random.uniform() < 0.2:
                genotype.mutate_add_edge(self.dataset, self.innovation_map)

            # Edges/Weights mutations
            for edge in genotype.edges:

                # Perturbations for genomes not created by DE
                if edge.mutable:
                    if np.random.uniform() < 0.95:
                        edge.mutate_perturbate_weight()
                    else:
                        edge.mutate_random_weight()

                # Smaller mutation for genomes created by DE
                else:
                    if np.random.uniform() < 0.03:
                        edge.mutate_shift_weight()
                    elif np.random.uniform() < 0.01:
                        edge.mutate_random_weight()

                # Reenabling
                if np.random.uniform() < 0.01 and not edge.enabled:
                    edge.enabled = True

    def new_evaluate_ancestors(self):
        Evaluator(self._population, self.dataset, True, self.seed).calculate_score()

    def new_evaluate(self):
        Evaluator(self._offspring_population, self.dataset, True, self.seed).calculate_score()

    def merge(self):
        self._population = []

        for offspring in self._offspring_population:
            ancestor = offspring.ancestor

            if ancestor is None or ancestor in self._population:
                self._population.append(offspring)
            else:
                self._population.append(offspring if offspring.score > ancestor.score else ancestor)
