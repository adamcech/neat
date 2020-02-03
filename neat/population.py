import math
import random

import numpy as np


from typing import List

from neat.config import Config
from neat.encoding.archive import Archive
from neat.encoding.innovation_map import InnovationMap
from neat.evaluator import Evaluator
from neat.species import Species
from neat.encoding.genotype import Genotype


class Population:

    def __init__(self, config: Config):
        self._species_id_counter = -1

        self.config = config
        self.innovation_map = InnovationMap()
        self.archive = Archive(self.config)

        self._compatibility = []  # type: List[float]
        self.species = []  # type: List[Species]
        self.population_offsprings = []  # type: List[Genotype]
        self._offspring_mutable = []  # type: List[bool]

        self.seed = self.config.dataset.get_random_seed(1)

        self.population = Genotype.init_population(self.config, self.innovation_map)  # type: List[Genotype]
        self.evaluate_ancestors()

    def _next_species_id(self):
        self._species_id_counter += 1
        return self._species_id_counter

    def _get_species_slots(self) -> List[int]:
        if len(self.species) == 1:
            return [self.config.population_size]

        species_fitness = [species.fitness for species in self.species]
        new_slots = [math.floor(species.fitness / sum(species_fitness) * self.config.population_size) for species in self.species]
        return new_slots

    def _adjust_fitness(self):
        full_pop = self.population + self.population_offsprings

        m = min([genotype.score for genotype in full_pop])
        m = abs(m) if m < 0 else 0

        for genotype in full_pop:
            genotype.fitness = genotype.score + m

    def speciate(self):
        self._adjust_fitness()

        for species in self.species:
            species.reset()

        for genotype in self.population:
            species_found = False
            best_species = -1
            best_compatibility = 0
            for i in range(len(self.species)):
                is_compatible, compatibility = self.species[i].calculate_compatibility(genotype, self.config.compatibility_threshold, self.config.compatibility_max_diffs)
                if is_compatible:
                    species_found = True
                    if compatibility > best_compatibility:
                        best_compatibility = compatibility
                        best_species = i

            if species_found:
                self.species[best_species].add_member(genotype, best_compatibility)
            else:
                self.species.append(Species(genotype, self._next_species_id(), self.config.species_representative_change))

        self.species = [species for species in self.species if not species.is_empty()]

        # calculated with ALL members before removing worst
        for species in self.species:
            species.evaluate_fitness()

        # Removing worst members of species
        for species in self.species:
            species.remove_worst(self.config.species_remove)

        # Remove Extinct and Stagnating
        self.species = [species for species in self.species if not species.is_extinct()]

        # Limit max species
        if len(self.species) > self.config.species_max:
            species_fitness = np.argsort([s.fitness for s in self.species])[::-1]
            best_member_species = self.species[np.argsort([s.get_best_member().score for s in self.species])[-1]]
            new_species = [best_member_species]

            species_counter = 0
            while len(new_species) < self.config.species_max:
                if self.species[species_fitness[species_counter]] not in new_species:
                    new_species.append(self.species[species_fitness[species_counter]])
                species_counter += 1

            self.species = new_species

        # save compatibility
        self._compatibility = []
        for species in self.species:
            for genotype_compatibility in species.compatibility:
                self._compatibility.append(genotype_compatibility)

    def archivate(self):
        species_slots = self._get_species_slots()

        for i in range(len(self.species)):
            self.archive.add(self.species[i].get_members_over_slot_limit(species_slots[i]))

        for i in range(len(self.species)):
            diff = species_slots[i] - len(self.species[i].members)
            if diff > 0:
                self.species[i].add_templates(self.config, self.archive.get_by_compatibility(self.species[i].representative, diff))

    def crossover(self):
        self.population_offsprings = []
        self._offspring_mutable = []
        species_slots = self._get_species_slots()
        is_low_compatibility = self.get_compatibility() < (self.config.compatibility_threshold + ((1 - self.config.compatibility_threshold) * self.config.compatibility_low))

        for i in range(len(self.species)):
            species = self.species[i]
            slots = species_slots[i]
            species_elite = species.get_elite(self.config.species_elitism)

            for j in range(slots):
                ancestor = species.members[j % len(species.members)]

                mutable = (len(self.species) == self.config.species_max and is_low_compatibility) or np.random.uniform() < self.config.compatibility_low_crossover
                self._offspring_mutable.append(mutable)

                if mutable:
                    if np.random.uniform() < self.config.species_mating or len(self.species) == 1:
                        # Species mating
                        parents = random.sample(species.members, 2)
                    else:
                        # Interspecies mating
                        parents = random.sample(self.population, 2)

                    best = random.sample(species_elite, 1)[0]
                    mom = parents[0]
                    dad = parents[1]

                    offspring = Genotype.crossover_triple(self.config.crossover_f, ancestor, best, mom, dad)
                    self.population_offsprings.append(offspring)
                else:
                    offspring = Genotype.crossover_single(ancestor, self.config, self.innovation_map)
                    self.population_offsprings.append(offspring)

    def mutate(self):
        for genotype in [self.population_offsprings[i] for i in range(len(self.population_offsprings)) if self._offspring_mutable[i]]:

            # Disabling closest to zero
            enabled_edges = [edge for edge in genotype.edges if edge.enabled]
            if len(enabled_edges) > 0 and np.random.uniform() < self.config.mutate_disable_lowest:
                edge = enabled_edges[0]
                for i in range(1, len(enabled_edges)):
                    if abs(enabled_edges[i].weight) < abs(edge.weight):
                        edge = enabled_edges[i]
                edge.enabled = False

            # Topology mutations
            if np.random.uniform() < self.config.mutate_add_node:
                genotype.mutate_add_node(self.config, self.innovation_map)
            if np.random.uniform() < self.config.mutate_add_edge:
                genotype.mutate_add_edge(self.config, self.innovation_map)

            # Edges/Weights mutations
            for edge in genotype.edges:

                # Perturbations for genomes not created by DE
                if edge.mutable and edge.enabled:
                    if np.random.uniform() < self.config.mutate_nde_perturbation_over_random:
                        edge.mutate_perturbate_weight()
                    else:
                        edge.mutate_random_weight()

                # Smaller mutation for genomes created by DE
                if not edge.mutable and edge.enabled:
                    if np.random.uniform() < self.config.mutate_de_shift:
                        edge.mutate_shift_weight()
                    elif np.random.uniform() < self.config.mutate_de_random:
                        edge.mutate_random_weight()

                # Enabling
                if np.random.uniform() < self.config.mutate_enable and not edge.enabled:
                    edge.enabled = True

    def evaluate_ancestors(self):
        Evaluator(self.population, self.config, True, self.seed).calculate_score()

    def evaluate_offsprings(self):
        Evaluator(self.population_offsprings, self.config, True, self.seed).calculate_score()

    def merge(self):
        original_population = [g for g in self.population]
        self.population = []

        for offspring in self.population_offsprings:
            if offspring.ancestor is None or offspring.ancestor in self.population:
                self.population.append(offspring)
            else:
                self.population.append(offspring if offspring.score > offspring.ancestor.score else offspring.ancestor)

        self.archive.add([g for g in original_population if g not in self.population])

    def next_seed(self):
        best = self.population[np.argsort([g.score for g in self.population])[-1]]

        new_seed = Evaluator([best], self.config, True, self.config.dataset.get_random_seed(self.config.seed_attempts)).find_best_seed(self.config.seed_select)

        if new_seed is not None:
            if self.seed is None:
                self.seed = new_seed
            elif self.seed is not None:
                self.seed += new_seed

            self.seed = self.seed[-self.config.seed_max:]

        self.archive.clear()
        self.evaluate_ancestors()

    def get_compatibility(self) -> float:
        return sum(self._compatibility) / len(self._compatibility)

    def get_best_member(self) -> Genotype:
        best_score = 0
        for i in range(1, len(self.population)):
            if self.population[i].score > self.population[best_score].score:
                best_score = i
        return self.population[best_score]

    def get_worst_member(self) -> Genotype:
        worst_score = 0
        for i in range(1, len(self.population)):
            if self.population[i].score < self.population[worst_score].score:
                worst_score = i
        return self.population[worst_score]
