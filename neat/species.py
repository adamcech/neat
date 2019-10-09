import random

import numpy as np
from typing import List


class Species:
    """Genotype Species"""

    def __init__(self, representative: "Genotype"):
        self.representative = representative
        self.members = []  # type: List["Genotype"]
        self.members.append(representative)

        self._fitness = 0

    def add_member(self, genotype: "Genotype"):
        self.members.append(genotype)

    def reset(self):
        self.representative = random.sample(self.members, 1)[0]
        self.members.clear()

    def get_champ(self) -> "Genotype":
        return self.members[np.argsort([member.fitness for member in self.members])[-1]]

    def get_fitness(self) -> float:
        return self._fitness

    def evaluate_fitness(self):
        if len(self.members) == 0:
            self._fitness = 0
        else:
            self._fitness = sum(member.fitness for member in self.members) / len(self.members)

    def remove_worst(self, percentage: float):
        sort_indices = np.argsort([member.fitness for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(int(len(self.members) * (1 - percentage)))]

    def __repr__(self):
        return "### Fitness: " + str(int(self.get_fitness())) + ", Members: " + str(len(self.members)) + " ###"
