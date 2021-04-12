import random
from typing import List, Tuple

from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.strategies.population_strategies import PopulationStrategies


class PopulationStrategy:

    def __init__(self, config: Config, strategy: PopulationStrategies, init_prop: float):
        self.config = config

        self.success_history = []  # type: List[int]
        self.fails_history = []  # type: List[int]

        self.prop = init_prop
        self.strategy = strategy

        self.success = 0
        self.fails = 0
        self.offsprings = []
        self.succ_offsprings = []  # type: List[int]

    def generation_start(self):
        self.success = 0
        self.fails = 0
        self.offsprings = []
        self.succ_offsprings = []

        self.generation_start_impl()

    def generation_start_impl(self):
        raise NotImplementedError()

    def generation_end(self, succ_offsprings: List[Genotype]):
        for i in range(len(self.offsprings)):
            if self.offsprings[i] in succ_offsprings:
                self.succ_offsprings.append(i)

        self.success = len(self.succ_offsprings)
        self.fails = len(self.offsprings) - self.success

        self.generation_end_impl()

        self.success_history.append(self.success)
        self.fails_history.append(self.fails)

    def generation_end_impl(self):
        raise NotImplementedError()

    def crossover(self, parents: List[Genotype]) -> Tuple[Genotype, float, float, PopulationStrategies]:
        offspring = self.crossover_impl(parents)
        self.offsprings.append(offspring[0])
        return offspring

    def crossover_impl(self, parents: List[Genotype]) -> Tuple[Genotype, float, float, PopulationStrategies]:
        raise NotImplementedError()

    def crossover_callback(self, data: Tuple):
        raise NotImplementedError()

    def get_strategy(self) -> PopulationStrategies:
        return self.strategy

    def is_strategy(self, other: PopulationStrategies) -> bool:
        return self.strategy == other

    def __str__(self):
        return str(round(self.prop, 4)) + "    " + str(self.strategy)
