import math
import random
import time

import numpy as np

from typing import List, Tuple

import ray

from evaluators.evaluator import Evaluator
from evaluators.local_evaluator import LocalEvaluator
from evaluators.ray_evaluator import RayEvaluator
from neat.config import Config
from neat.encoding.archive import Archive
from neat.encoding.innovation_map import InnovationMap
from neat.strategies.mutation_manager import MutationManager
from neat.strategies.crossover_manager import CrossoverManager
from neat.species import Species
from neat.encoding.genotype import Genotype

import multiprocessing as mp

from neat.strategies.population_strategies import PopulationStrategies


class Population:

    def __init__(self, config: Config):
        self._species_id_counter = -1

        self.reached_max_species_generation = 0
        self.reached_max_species = False

        self.config = config
        self.innovation_map = InnovationMap()

        self.archive = Archive(self.config)

        self.compatibility = []  # type: List[float]
        self.species = []  # type: List[Species]
        self.population_offsprings = []  # type: List[Tuple[Genotype, Species, PopulationStrategies]]
        self.succ_offsprings = []  # type: List[Genotype]

        # self.test_seed = [i for i in range(self.config.seed_test)]  # self.config.dataset.get_random_seed(self.config.seed_test)
        self.test_seed = self.config.dataset.get_random_seed(self.config.seed_test)

        # self.seed = random.sample(self.test_seed, self.config.seed_max)
        self.seed = self.config.dataset.get_random_seed(self.config.seed_max)
        # self.seed = [5320921]
        # self.seed = [42]

        self.last_merge = 0

        self.crossover_manager = CrossoverManager(config)
        self.mutation_manager = MutationManager(config, self.innovation_map)

        self.population = Genotype.init_population(self.config, self.innovation_map)  # type: List[Genotype]

        self.evaluate_ancestors()

        self.comp_before = 1
        self.comp_after = 1

        self.specs_before = 0
        self.specs_after = 0

        self.stagnating = 0

        self.avg_score = []
        self._next_growth = 10

    def get_evaluator(self) -> Evaluator:
        return RayEvaluator(self.config) if self.config.cluster_evaluation else LocalEvaluator(self.config)

    def _next_species_id(self):
        self._species_id_counter += 1
        return self._species_id_counter

    def _get_species_slots(self) -> List[int]:
        if len(self.species) == 1:
            return [self.config.population_size]
        """
        species_fitness = [species.fitness for species in self.species]
        new_slots = [math.floor(species.fitness / sum(species_fitness) * self.config.population_size) for species in self.species]
        return new_slots
        """
        species_size = self._get_species_size()
        slots = [species_size for _ in range(len(self.species))]

        species_counter = 0
        while sum(slots) < self.config.population_size:
            slots[species_counter] += 1
            species_counter += 1

        return slots

    def _get_species_size(self) -> int:
        return int(self.config.population_size / len(self.species))

    def _adjust_fitness(self):
        full_pop = self.population + [g for g, _, _ in self.population_offsprings] + self.archive.archive
        full_pop += [g.ancestor for g in full_pop if g.ancestor is not None]

        m = min([genotype.score for genotype in full_pop])
        m = (abs(m) if m < 0 else 0) + 0.00001

        for genotype in full_pop:
            genotype.fitness = genotype.score + m

    def adapt_params_start(self):
        self.succ_offsprings = []
        self.crossover_manager.generation_start()
        self.stagnating = 0

    def adapt_params_end(self):
        self.crossover_manager.generation_end(self.succ_offsprings)
        self.mutation_manager.generation_end()

        avg_score = sum(g.score for g in self.population) / len(self.population)
        self.avg_score.append(avg_score)

    def _get_best_genotype_tested(self):
        pop_test = self.get_elite()
        _, avg_results, _ = self.get_evaluator().test(pop_test, self.test_seed)
        return pop_test[avg_results.index(max(avg_results))]

    def _get_interspecies_disjoints(self) -> float:
        if len(self.species) == 1:
            return 1
        else:
            representatives = [s.representative for s in self.species]
            compats = []
            for i in range(len(representatives)):
                c = []
                for j in range(len(representatives)):
                    c.append(representatives[i].is_compatible(representatives[j], self.config.compatibility_max_diffs)[1])
                compats.append(c)

            avg_compats = [sum([compats[i][j] for j in range(len(compats[i])) if j != i]) / (len(compats) - 1) for i in range(len(compats))]
            return sum(avg_compats) / len(avg_compats)

    def speciate(self):
        self.archive.reset_stats()
        self._adjust_fitness()

        for species in self.species:
            species.reset(self.species, self.population)

        self.species = [s for s in self.species if s.representative is not None]
        representatives = [s.representative for s in self.species if s.representative is not None]

        for genotype in self.population:
            if genotype in representatives:
                continue

            genotype.curr_orig_species_id = genotype.species_id
            species_found = False
            best_species = -1
            best_diffs = None
            for i in range(len(self.species)):
                is_compatible, diffs = self.species[i].calculate_compatibility(genotype)
                if is_compatible:
                    species_found = True
                    if best_diffs is None or diffs < best_diffs:
                        best_diffs = diffs
                        best_species = i

            if species_found:
                self.species[best_species].add_member(genotype, best_diffs)
            else:
                self.species.append(
                    Species(genotype, self._next_species_id(), self.config.compatibility_max_diffs, self.config.species_elitism))

        self.species = [species for species in self.species if not species.is_empty()]

        # For Console Observer
        self.specs_before = len(self.species)
        self.comp_before = self._get_interspecies_disjoints()

        for species in self.species:
            species.evaluate_fitness()

        best_member_species = self.species[np.argsort([s.get_best_member().score for s in self.species])[-1]]
        best_member = best_member_species.get_best_member()
        print("BEST MEMBER", best_member, best_member_species)

        # Remove Extinct
        # if len(self.species) >= self.config.species_max:
        #     self.species = [species for species in self.species if not species.is_extinct() or species == best_member_species]

        # Limit max species
        new_species = [best_member_species]  # type: List[Species]
        # species_score = np.argsort([s.fitness * best_member.calculate_compatibility(s.representative, self.config.compatibility_max_diffs) for s in self.species])[::-1]

        # if len(self.species) < self.config.species_max:
        #     valid_species = [s for s in self.species]
        # else:
        #     valid_species = [s for s in self.species if
        #                      s.calculate_compatibility(best_member_species.representative)[1] < self.config.compatibility_max_diffs * 3 and
        #                      s.get_best_member().fitness >= best_member_species.get_best_member().fitness * 0.5]

        valid_species = [s for s in self.species if
                         s.calculate_compatibility(best_member_species.representative)[1] < self.config.compatibility_species_max_diffs and
                         s.get_best_member().fitness >= best_member_species.get_best_member().fitness * 0.75]

        species_score = np.argsort([s.get_best_member().fitness + s.fitness for s in valid_species])[::-1]

        for i in species_score:
            if self.species[i] not in new_species:
                new_species.append(self.species[i])

            if len(new_species) == self.config.species_max:
                break

        self.species = new_species

        speices_id = {s.id for s in self.species}
        unspeciated_genotypes = [g for g in self.population if g.species_id not in speices_id]

        for genotype in unspeciated_genotypes:
            genotype.species_id = genotype.curr_orig_species_id
            species_found = False
            best_species = None
            best_diffs = None
            for i in range(len(self.species)):
                is_compatible, diffs = self.species[i].calculate_compatibility(genotype)
                if is_compatible:
                    species_found = True
                    if best_species is None or diffs < best_diffs:
                        best_diffs = diffs
                        best_species = i
            if species_found:
                self.species[best_species].add_member(genotype, best_diffs)

        # if len(self.species) >= self.config.species_max and self._get_interspecies_disjoints() > self.config.compatibility_species_max_diffs:

        # if len(self.species) >= self.config.species_max:
        #     representatives = [s.representative for s in self.species]
        #     representatives_sizes = [len(r.edges) for r in representatives]
        #     avg_size = sum(representatives_sizes) / len(representatives_sizes)
        #
        #     best_member_species = self.species[np.argsort([s.get_best_member().score for s in self.species])[-1]]
        #
        #     self.species = [s for s in self.species
        #                     if s.calculate_compatibility(best_member_species.representative)[1] < self.config.compatibility_max_diffs * 3]

            # if self._get_interspecies_disjoints() > max(avg_size * 0.2, self.config.compatibility_max_diffs * 3):
            #     pop_test = []
            #     for s in self.species:
            #         pop_test.extend(s.get_elite())
            #
            #     results, avg_results, seeds = self.get_evaluator().test(pop_test, self.test_seed, evals=True, allow_penalty=False)
            #
            #     avg_results = [avg_results[i] + pop_test[i].score for i in range(len(pop_test))]
            #     best_genotype = pop_test[avg_results.index(max(avg_results))]
            #
            #     best_member_species = Species(best_genotype, self._next_species_id(), self.config.compatibility_max_diffs, self.config.species_elitism)
            #
            #     self.species = [best_member_species]
            #     self.archive.add(self.population)
            #     best_member_species.restart_to_best(self.config, best_member_species.get_best_member(), self.population)
            #     self.get_evaluator().calculate_score(self.population, self.seed)
            #     self._adjust_fitness()
            #     self.mutaiton_manager.after_merge = True

        for species in self.species:
            species.evaluate_fitness()
            species.sort_members_by_score()

        speciated_genotypes = []
        [speciated_genotypes.append(g) for s in self.species for g in s.members if g not in speciated_genotypes]

        # Add unspeciated genotypes
        self.archive.add([g for g in self.population if g not in speciated_genotypes])

        # For Console Observer
        self.compatibility = [g.species_diff for s in self.species for g in s.members]
        self.comp_after = self._get_interspecies_disjoints()
        self.specs_after = len(self.species)

    def archivate(self):
        self._adjust_fitness()
        species_slots = self._get_species_slots()

        for i in range(len(self.species)):
            self.archive.add(self.species[i].get_members_over_slot_limit(species_slots[i]))

        returned_genotypes = []
        # Add similar by compatibility
        for i in range(len(self.species)):
            diff = species_slots[i] - len(self.species[i].members)
            if diff > 0:
                from_archive = self.archive.get_by_compatibility(self.species[i].representative, diff)
                returned_genotypes.extend(from_archive)
                self.species[i].add_templates(self.config, from_archive, False, self.innovation_map)
        # Add rest
        for i in range(len(self.species)):
            diff = species_slots[i] - len(self.species[i].members)
            if diff > 0:
                from_archive = self.archive.get_by_highest_score(diff)
                returned_genotypes.extend(from_archive)
                self.species[i].add_templates(self.config, from_archive, False, self.innovation_map)

        stagnating_lst = []
        for species in self.species:
            stagnating_lst.extend(species.mutate_stagnating_members(self.config, self.innovation_map))
        self.stagnating = len(stagnating_lst)

        corrected_genotypes = []
        [corrected_genotypes.append(g) for g in returned_genotypes + stagnating_lst if g not in corrected_genotypes]

        self.get_evaluator().calculate_score(corrected_genotypes, self.seed)

    def crossover(self):
        self._adjust_fitness()
        for s in self.species:
            s.sort_members_by_score()

        self.population_offsprings = []

        species_slots = self._get_species_slots()
        elite = self.get_elite()

        for i in range(len(self.species)):
            slots = species_slots[i]
            species = self.species[i]
            species_elite = species.get_elite()

            for j in range(slots):
                ancestor = species.roulette_select(species_elite, 1)[0] if j >= int((0.9 * len(species.members))) else species.members[j]

                offspring, f, cr, strategy = self.crossover_manager.get_offspring(
                    ancestor, self.population, elite, species.members, species_elite, self.config)

                if int((0.9 * len(species.members))) <= j < len(species.members):
                    offspring.ancestor = species.members[j]

                self.population_offsprings.append((offspring, species, strategy))
                if strategy == PopulationStrategies.SKIP:
                    self.stagnating += 1

    def mutate_topology(self):
        self.mutation_manager.restart(self.species)

        for offspring, species, strategy in self.population_offsprings:
            self.mutation_manager.mutate(offspring, species)

    def evaluate_ancestors(self):
        self.get_evaluator().calculate_score(self.population, self.seed)

    def evaluate_offsprings(self):
        self.get_evaluator().calculate_score([g for g, _, _ in self.population_offsprings], self.seed)

    def merge(self):
        self._adjust_fitness()

        elite = [self.get_best_member()]
        for s in self.species:
            elite.extend(s.get_elite())

        archive_appends = []
        self.population = []

        for offspring, _, strategy in self.population_offsprings:
            append_offspring = offspring.is_better() or \
                               offspring.ancestor in self.population or \
                               strategy == PopulationStrategies.SKIP  # or (np.random.uniform() < 0.03 and (offspring.ancestor is not None and offspring.ancestor not in elite))

            if offspring.ancestor is not None and offspring.ancestor in elite and offspring.ancestor.score >= offspring.score:
                append_offspring = False

            self.population.append(offspring if append_offspring else offspring.ancestor)

            if append_offspring:
                self.succ_offsprings.append(offspring)
                if np.random.uniform() < self.config.archivate_prop:
                    archive_appends.append(offspring.ancestor)

        self.archive.add(archive_appends)
        self.archive.remove_old()

    def next_seed(self) -> float:
        seed_type = self.config.dataset.get_seed_type()

        if seed_type is None:
            return self.next_seed_none()
        elif seed_type is list:
            return self.next_seed_list()
        else:
            raise Exception("Unknown seed type")

    def next_seed_none(self) -> float:
        elite_scores = []
        for s in self.species:
            for g in s.get_elite():
                elite_scores.append(g.score)
        return sum(elite_scores) / len(elite_scores)

    def next_seed_list(self) -> float:
        new_seed = []
        test_score = None

        best_genotype = self.get_best_member()
        best_pop = [s.get_best_member() for s in self.species]
        if best_genotype not in best_pop:
            best_pop.append(best_genotype)

        # for s in self.species:
        #     species_elite = s.get_elite()
        #     for genotype in species_elite:
        #         if genotype not in best_pop:
        #             best_pop.append(genotype)

        if self.config.seed_select_random >= 1:
            new_seed += self.config.dataset.get_random_seed(self.config.seed_select_random)

        if self.config.seed_select >= 1:
            if self.config.seed_attempts == 0:
                new_seed += self.config.dataset.get_random_seed(self.config.seed_select)
            else:
                generated_seed, test_score = self.get_evaluator().calculate_best_avg_seed(self.config.seed_select, best_pop, self.config.dataset.get_random_seed(self.config.seed_attempts))
                new_seed += generated_seed

        self.seed += new_seed
        self.seed = self.seed[-self.config.seed_max:]

        pop = self.archive.archive + self.population
        [pop.append(g.worse_ancestor) for g in pop if g.worse_ancestor is not None]
        self.get_evaluator().calculate_score(pop, self.seed, forced=False)

        # self.get_evaluator().calculate_score(self.archive.archive, self.seed, forced=False)
        # self.get_evaluator().calculate_score(self.population, self.seed, forced=False)

        return test_score

    def test(self, test_size: int = 10) -> Tuple[float, float, float, Genotype]:
        pop_score_indices = np.argsort([g.score for g in self.population])[::-1]
        pop_test = [self.population[pop_score_indices[i]] for i in range(test_size)]
        results, avg_results, seeds = self.get_evaluator().test(pop_test, self.test_seed, evals=False, allow_penalty=False)
        return min(avg_results), sum(avg_results) / len(avg_results), max(avg_results), pop_test[avg_results.index(max(avg_results))]

    def get_compatibility(self) -> float:
        return sum(self.compatibility) / len(self.compatibility)

    def get_elite(self) -> List[Genotype]:
        sort_indices = np.argsort([g.score for g in self.population])[::-1]
        return [self.population[i] for i in sort_indices[:int(len(sort_indices) * self.config.species_elitism)]]

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
