import math
from typing import List, Tuple


from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.strategies.curr_to_best import CurrToBest
from neat.strategies.curr_to_rand import CurrToRand
from neat.strategies.rand_to_best import RandToBest
from neat.strategies.rand_to_rand import RandToRand
from neat.strategies.population_strategies import PopulationStrategies
from neat.strategies.population_strategy import PopulationStrategy

import numpy as np


class CrossoverManager:

    def __init__(self, config: Config):
        self.config = config
        self.min_add = math.ceil(config.population_size * 0.1)

        self.curr_to_best_spec = CurrToBest(self.config, PopulationStrategies.CURR_TO_BEST_SPEC, 0.25)
        self.curr_to_best_interspec = CurrToBest(self.config, PopulationStrategies.CURR_TO_BEST_INTERSPEC, 0.25)

        self.curr_to_rand_spec = CurrToRand(self.config, PopulationStrategies.CURR_TO_RAND_SPEC, 0.25)
        self.curr_to_rand_interspec = CurrToRand(self.config, PopulationStrategies.CURR_TO_RAND_INTERSPEC, 0.25)

        self.rand_to_rand_spec = RandToRand(self.config, PopulationStrategies.RAND_TO_RAND_SPEC, 0.25)
        self.rand_to_rand_interspec = RandToRand(self.config, PopulationStrategies.RAND_TO_RAND_INTERSPEC, 0.25)

        self.rand_to_best_spec = RandToBest(self.config, PopulationStrategies.RAND_TO_BEST_SPEC, 0.25)
        self.rand_to_best_interspec = RandToBest(self.config, PopulationStrategies.RAND_TO_BEST_INTERSPEC, 0.25)

        self.strategies = [self.curr_to_best_spec, self.curr_to_best_interspec, self.rand_to_rand_spec, self.rand_to_rand_interspec]
        # self.strategies = [self.rand_to_best_spec, self.rand_to_best_interspec, self.rand_to_rand_spec, self.rand_to_rand_interspec]
        # self.strategies = [self.curr_to_best_spec, self.curr_to_best_interspec, self.curr_to_rand_spec, self.curr_to_rand_interspec]

    def reset(self):
        for s in self.strategies:
            s.prop = 1 / len(self.strategies)

    def get_offspring_callback(self, genotype: Genotype, f: float, cr: float, strategy: PopulationStrategies):
        self.get_strategy_by_id(strategy).crossover_callback((genotype, f, cr))

    def get_strategy_by_id(self, strategy: PopulationStrategies) -> PopulationStrategy:
        if strategy == PopulationStrategies.CURR_TO_BEST_SPEC:
            return self.curr_to_best_spec
        elif strategy == PopulationStrategies.CURR_TO_BEST_INTERSPEC:
            return self.curr_to_best_interspec
        if strategy == PopulationStrategies.CURR_TO_RAND_SPEC:
            return self.curr_to_rand_spec
        elif strategy == PopulationStrategies.CURR_TO_RAND_INTERSPEC:
            return self.curr_to_rand_interspec
        elif strategy == PopulationStrategies.RAND_TO_RAND_SPEC:
            return self.rand_to_rand_spec
        elif strategy == PopulationStrategies.RAND_TO_RAND_INTERSPEC:
            return self.rand_to_rand_interspec
        elif strategy == PopulationStrategies.RAND_TO_BEST_SPEC:
            return self.rand_to_best_spec
        elif strategy == PopulationStrategies.RAND_TO_BEST_INTERSPEC:
            return self.rand_to_best_interspec

    def get_offspring(self, ancestor: Genotype, population: List[Genotype], elite: List[Genotype], species: List[Genotype], species_elite: List[Genotype], config: Config) -> Tuple[Genotype, float, float, PopulationStrategies]:
        if np.random.uniform() < 0.01:  # or config.generation - ancestor.origin_generation >= 5:
            return ancestor.mutation_copy(config.generation), 0.0, 0.0, PopulationStrategies.SKIP

        strategy = self.get_strategy()

        if strategy == self.curr_to_best_spec:
            best = self.roulette_select(species_elite, 1, [ancestor])[0]
            parents = self.roulette_select(species, 2, [best, ancestor])
            return self.curr_to_best_spec.crossover([ancestor, best, parents[0], parents[1]])

        elif strategy == self.curr_to_best_interspec:
            best = self.roulette_select(elite, 1, [ancestor])[0]
            parents = self.roulette_select(population, 2, [ancestor, best])
            return self.curr_to_best_interspec.crossover([ancestor, best, parents[0], parents[1]])

        elif strategy == self.curr_to_rand_spec:
            parents = self.roulette_select(species, 3, [ancestor])
            return self.curr_to_rand_spec.crossover([ancestor, parents[0], parents[1], parents[2]])

        elif strategy == self.curr_to_rand_interspec:
            parents = self.roulette_select(population, 3, [ancestor])
            return self.curr_to_rand_interspec.crossover([ancestor, parents[0], parents[1], parents[2]])

        elif strategy == self.rand_to_rand_spec:
            parents = self.roulette_select(species, 3)
            return self.rand_to_rand_spec.crossover([ancestor, parents[0], parents[1], parents[2]])

        elif strategy == self.rand_to_rand_interspec:
            parents = self.roulette_select(population, 3)
            return self.rand_to_rand_interspec.crossover([ancestor, parents[0], parents[1], parents[2]])

        elif strategy == self.rand_to_best_spec:
            best = self.roulette_select(species_elite, 1)[0]
            parents = self.roulette_select(species, 2, [best])
            return self.rand_to_best_spec.crossover([ancestor, best, parents[0], parents[1]])

        elif strategy == self.rand_to_best_interspec:
            best = self.roulette_select(elite, 1)[0]
            parents = self.roulette_select(population, 2, [best])
            return self.rand_to_best_interspec.crossover([ancestor, best, parents[0], parents[1]])

    def get_strategy(self) -> PopulationStrategy:
        r = np.random.uniform()

        for s in self.strategies:
            if r < s.prop:
                return s
            else:
                r -= s.prop

    def roulette_select(self, population: List[Genotype], count: int, exclude: List[Genotype] = None) -> List[Genotype]:
        selected = []
        population = list(population)

        if exclude is not None:
            for g in exclude:
                if g in population:
                    population.remove(g)

        for i in range(count):
            g = self.sample_roulette(population)
            selected.append(g)

            if g not in population:
                print()
            population.remove(g)

        return selected

    def sample_roulette(self, population: List[Genotype]) -> Genotype:
        total = sum(g.fitness for g in population)
        if total > 0:
            props = [g.fitness / total for g in population]
        else:
            props = [1 / len(population) for g in population]

        r = np.random.uniform()

        for i in range(len(population)):
            r -= props[i]
            if r <= 0:
                return population[i]

    def generation_start(self):
        for s in self.strategies:
            s.success_history = []
            s.fails_history = []
            s.generation_start()

    def generation_end(self, succ_offsprings: List[Genotype]):
        for s in self.strategies:
            s.generation_end(succ_offsprings)

        if self.config.generation % self.config.learning_period == 0 and self.config.generation != 0 and self.config.learning_period >= 1:
            ss = [sum(s.success_history[-self.config.learning_period:]) + self.min_add for s in self.strategies]
            sf = [sum(s.fails_history[-self.config.learning_period:]) + self.min_add for s in self.strategies]

            all_sum = 0.0
            for i in range(len(self.strategies)):
                all_sum += ss[i] * (sum(ss[j] for j in range(len(ss)) if j != i) + sum(sf[j] for j in range(len(sf)) if j != i))

            if all_sum == 0.0:
                for i in range(len(self.strategies)):
                    self.strategies[i].prop = 1 / len(self.strategies)
            else:
                for i in range(len(self.strategies)):
                    self.strategies[i].prop = (ss[i] * (sum(ss[j] for j in range(len(ss)) if j != i) + sum(sf[j] for j in range(len(sf)) if j != i))) / all_sum
