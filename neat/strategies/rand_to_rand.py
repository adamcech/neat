from typing import List, Tuple

from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.strategies.population_strategies import PopulationStrategies
from neat.strategies.population_strategy import PopulationStrategy

import numpy as np


class RandToRand(PopulationStrategy):

    def __init__(self, config: Config, strategy: PopulationStrategies, init_prop: float):
        super().__init__(config, strategy, init_prop)

        self.cr = []
        self.f = []

    def generate_mutation_factor_norm(self) -> float:
        f = np.random.normal(0.5, 0.3)
        return 0.1 if f <= 0.1 else f

    def crossover_impl(self, parents: List[Genotype]) -> Tuple[Genotype, float, float, PopulationStrategies]:
        cr = np.random.normal(0.9, 0.1)
        cr = 1.0 if cr > 1.0 else (0.1 if cr < 0.1 else cr)
        f = self.generate_mutation_factor_norm()

        self.cr.append(cr)
        self.f.append(self.generate_mutation_factor_norm())

        return Genotype.crossover(self.config.generation, f, cr, parents[0], parents[1], parents[2], parents[3]), f, cr, self.strategy

    def crossover_callback(self, data: Tuple[Genotype, float, float]):
        genotype, f, cr = data

        self.f.append(f)
        self.cr.append(cr)
        self.offsprings.append(genotype)

    def generation_start_impl(self):
        self.f = []
        self.cr = []

    def generation_end_impl(self):
        pass

    def __str__(self):
        return self.strategy.get_name() + "; p = " + str(round(self.prop, 2))
