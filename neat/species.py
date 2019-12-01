import math
import random

import numpy as np
from typing import List

from neat.encoding.genotype import Genotype


class Species:
    """Genotype Species"""

    __stagnation_time = 20
    __extinct_size = 2
    __dying_size = 5

    __id_counter = 1

    def __init__(self, representative: "Genotype"):
        self.id = Species.__id_counter
        Species.__id_counter += 1

        self.representative = representative
        self.members = []  # type: List["Genotype"]
        self.compatibility = []  # type: List[float]
        self.members.append(representative)

        self.score_history = []
        self.fitness = 0
        self.score = 0

        self._best_member = None

    def add_member(self, genotype: "Genotype", compatibility: float):
        self.members.append(genotype)
        self.compatibility.append(compatibility)

    def reset(self):
        self._best_member = None
        self.score_history.append(max([genotype.score for genotype in self.members]))
        self.score_history = self.score_history[-self.__stagnation_time:]

        self.representative = random.sample(self.members, 1)[0]
        self.members.clear()
        self.compatibility.clear()

    def union(self, other: "Species"):
        self._best_member = self.get_best_member() if self.get_best_member().score >= other.get_best_member().score else other.get_best_member()
        self.members += other.members
        self.compatibility += other.compatibility
        self.score_history = np.average(np.array([self.score_history, other.score_history]), axis=0)

    def get_best_member(self) -> "Genotype":
        if self._best_member is None:
            self._best_member = self.members[np.argsort([member.fitness for member in self.members])[-1]]
        return self._best_member

    def evaluate_fitness(self):
        self.fitness = 0 if len(self.members) == 0 else sum(member.fitness for member in self.members) / len(self.members)
        self.score = 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def remove_worst(self, percentage: float):
        sort_indices = np.argsort([member.fitness for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(int(len(self.members) * (1 - percentage)))]

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def is_extinct(self) -> bool:
        return len(self.members) <= Species.__extinct_size

    def is_dying(self):
        return len(self.members) <= Species.__dying_size

    def is_mature(self):
        return len(self.score_history) == self.__stagnation_time

    def is_stagnating(self) -> bool:
        return max(self.score_history) == self.score_history[0] if self.is_mature() else False

    def __repr__(self):
        return "### Fitness: " + str(self.score) + ", Members: " + str(len(self.members)) + " ###"

    def get_elite(self, elitism: int) -> List[Genotype]:
        return [self.members[i] for i in range(min(elitism, len(self.members)))]
