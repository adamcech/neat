import math
import random
from typing import List

import numpy as np

from neat.config import Config


class Edge:
    """Genome representation of connection genes
    """
    def __init__(self, genotype, config: Config, input_node: int, output_node: int, enabled: bool, innovation: int, mutable: bool = False, weight: float = None):
        self.config = config
        self.genotype = genotype

        self.input = input_node
        self.output = output_node
        self.enabled = enabled
        self.innovation = innovation

        self.mutable = mutable

        self.weight = weight
        self.mutate_random_weight() if self.weight is None else self.set_weight(self.weight)

    def get_output_size(self) -> int:
        input_size = sum(1 for e in self.genotype.edges if e.output == self.output and e.enabled)
        if self not in self.genotype.edges:
            input_size += 1
        input_size = max(input_size, 1)
        return input_size

    def get_output_size_scale(self) -> float:
        input_size = sum(1 for e in self.genotype.edges if e.output == self.output and e.enabled)
        if self not in self.genotype.edges:
            input_size += 1
        input_size = max(input_size, 1)
        return math.sqrt(2 / input_size)

    def mutate_random_weight(self):
        self.set_weight(np.random.normal(0.0, random.sample(self.config.weight_random_scale, 1)[0]) * self.get_output_size_scale())

    def mutate_perturbate_weight(self):
        self.set_weight(self.weight + np.random.normal(0.0, random.sample(self.config.weight_pert_scale, 1)[0]) * self.get_output_size_scale())

    def mutate_shift_weight(self):
        self.set_weight(self.weight * np.random.uniform(self.config.mutate_shift_weight_lower_bound, self.config.mutate_shift_weight_upper_bound))

    def set_weight(self, weight: float):
        if abs(weight) > self.config.weight_scale:
            self.weight = (self.config.weight_scale if weight > 0 else -self.config.weight_scale) * np.random.uniform(0.95, 1.0)
        else:
            self.weight = weight

    def __repr__(self):
        return str(self.input) + "->" + str(self.output) + " (" + str(self.innovation) + ") " + str(self.weight) if self.enabled else ""

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
