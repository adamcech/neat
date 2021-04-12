import random
from typing import Union, List

from neat.config import Config
from neat.encoding.genotype import Genotype

import numpy as np


class Archive:

    def __init__(self, config: Config):
        self.config = config
        self._archive_size = int(config.population_size)
        self.archive = []  # type: List[Genotype]

        self.additions = 0
        self.removes = 0

    def get_genotypes(self) -> List[Genotype]:
        return self.archive

    def clear(self):
        self.archive.clear()
        self.reset_stats()

    def add(self, g: Union[List[Genotype], Genotype]):
        if type(g) == Genotype:
            g = [g]

        for genotype in g:
            if genotype not in self.archive:
                self.additions += 1
                self.archive.append(genotype.copy())

    def remove_old(self):
        if len(self.archive) > self._archive_size:
            # scores_indices = np.argsort([g.score for g in self.archive])[::-1]
            # self.archive = [self.archive[scores_indices[i]] for i in range(self._archive_size)]
            self.archive = self.archive[-self._archive_size:]

    def get_by_compatibility(self, representative: Genotype, count: int) -> List[Genotype]:
        if count <= 0:
            return []

        compatible = []
        uncompatible = []

        for g in self.archive:
            if representative.is_compatible(g, self.config.compatibility_max_diffs)[0]:
                compatible.append(g)
            else:
                uncompatible.append(g)

        compatible_indices = np.argsort([g.score for g in compatible])[::-1]
        candidates = [compatible[compatible_indices[i]] for i in range(min(len(compatible_indices), count))]

        for c in candidates:
            self.archive.remove(c)

        self.removes += len(candidates)
        return candidates

    def get_by_highest_score(self, count: int) -> List[Genotype]:
        if count <= 0:
            return []

        score_indices = np.argsort([g.score for g in self.archive])[::-1]
        candidates = [self.archive[score_indices[i]] for i in range(min(count, len(score_indices)))]

        for c in candidates:
            self.archive.remove(c)

        self.removes += len(candidates)
        return candidates

    def get_random(self, count: int):
        if count <= 0:
            return []

        candidates = random.sample(self.archive, count)  # type: List[Genotype]

        for c in candidates:
            self.archive.remove(c)

        self.removes += len(candidates)
        return candidates

    def reset_stats(self) -> None:
        self.additions, self.removes = 0, 0

    def get_info(self):
        return str(len(self.archive)).ljust(3) + " +" + str(self.additions).ljust(3) + " -" + str(self.removes).ljust(3)
