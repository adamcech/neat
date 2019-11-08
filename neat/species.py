import math
import random

import numpy as np
from typing import List


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

    def add_member(self, genotype: "Genotype", compatibility: float):
        self.members.append(genotype)
        self.compatibility.append(compatibility)

    def reset(self):
        # self.score_history.append(self.score)
        self.score_history.append(max([genotype.score for genotype in self.members]))
        self.score_history = self.score_history[-self.__stagnation_time:]

        self.representative = random.sample(self.members, 1)[0]
        self.members.clear()
        self.compatibility.clear()

    def get_champ(self) -> "Genotype":
        return self.members[np.argsort([member.fitness for member in self.members])[-1]]

    def evaluate_fitness(self):
        self.fitness = 0 if len(self.members) == 0 else sum(member.fitness for member in self.members) / len(self.members)
        self.score = 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def remove_worst(self, percentage: float):
        sort_indices = np.argsort([member.fitness for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(int(len(self.members) * (1 - percentage)))]

    def is_extinct(self) -> bool:
        return len(self.members) <= Species.__extinct_size

    def is_dying(self):
        return len(self.members) <= Species.__dying_size

    def is_stagnating(self) -> bool:
        if len(self.score_history) == self.__stagnation_time:
            if max(self.score_history) == self.score_history[0]:
                print("Species stagnation")
                return True
        return False

    def __repr__(self):
        return "### Fitness: " + str(self.score) + ", Members: " + str(len(self.members)) + " ###"

    def get_elite(self, elitism: float) -> List["Genotype"]:
        return [self.members[i] for i in range(math.ceil(len(self.members) * elitism))]
