from typing import List, Tuple

from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.strategies.population_strategies import PopulationStrategies
from neat.strategies.population_strategy import PopulationStrategy

import numpy as np


class CurrToBest(PopulationStrategy):

    def __init__(self, config: Config, strategy: PopulationStrategies, init_prop: float):
        super().__init__(config, strategy, init_prop)

        self.cr = []
        self.f = []

        self.f_loc_cauchy = 0.7
        self.cr_loc = 0.9

    def generate_cr(self) -> float:
        cr = np.random.normal(self.cr_loc, 0.1)
        return 1.0 if cr > 1.0 else (0.1 if cr <= 0.1 else cr)

    def _generate_mutation_factor_cauchy(self) -> float:
        f = np.random.standard_cauchy() * 0.1 + self.f_loc_cauchy
        return 1.0 if f > 1.0 else (f if f > 0.1 else 0.1)

    def crossover_impl(self, parents: List[Genotype]) -> Tuple[Genotype, float, float, PopulationStrategies]:
        cr = self.generate_cr()
        f = np.random.uniform() if np.random.uniform() < 0.2 else self._generate_mutation_factor_cauchy()

        self.cr.append(cr)
        self.f.append(f)

        return Genotype.crossover_triple(self.config.generation, f, cr, parents[0], parents[1], parents[2], parents[3]), f, cr, self.strategy

    def crossover_callback(self, data: Tuple[Genotype, float, float]):
        genotype, f, cr = data

        self.f.append(f)
        self.cr.append(cr)
        self.offsprings.append(genotype)

    def generation_start_impl(self):
        self.cr = []
        self.f = []

    def generation_end_impl(self):
        mutation_factors = [self.f[i] for i in self.succ_offsprings]
        cr_factors = [self.cr[i] for i in self.succ_offsprings]

        if len(mutation_factors) >= 2:
            self.f_loc_cauchy = (1 - self.config.jade_c) * self.f_loc_cauchy + \
                                          self.config.jade_c * sum([x ** 2 for x in mutation_factors]) / sum(mutation_factors)

        if len(cr_factors) >= 2:
            self.cr_loc = (1 - self.config.jade_c) * self.cr_loc + self.config.jade_c * sum(cr_factors) / len(cr_factors)

    def __str__(self):
        return self.strategy.get_name() + "; p = " + str(round(self.prop, 2)) + "; f = " + str(round(self.f_loc_cauchy, 2)) + "; cr = " + str(round(self.cr_loc, 2))
