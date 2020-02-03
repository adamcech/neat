from typing import Union, List

from neat.config import Config
from neat.encoding.genotype import Genotype

import numpy as np


class Archive:

    def __init__(self, config: Config):
        self.config = config
        self._archive_size = config.population_size * 20
        self._archive = []  # type: List[Genotype]

    def clear(self):
        self._archive.clear()

    def add(self, g: Union[List[Genotype], Genotype]):
        if type(g) == Genotype:
            g = [g]

        for genotype in g:
            if genotype not in self._archive:
                genotype.ancestor = None
                self._archive.append(genotype)

        if len(self._archive) > self._archive_size:
            scores_indices = np.argsort([g.score for g in self._archive])[::-1]
            self._archive = [self._archive[scores_indices[i]] for i in range(self._archive_size)]

    def get_by_compatibility(self, representative: Genotype, count: int) -> List[Genotype]:
        if count <= 0:
            return []

        compatible = []
        uncompatible = []

        for g in self._archive:
            if representative.is_compatible(g, self.config.compatibility_threshold, self.config.compatibility_max_diffs)[0]:
                compatible.append(g)
            else:
                uncompatible.append(g)

        compatible_indices = np.argsort([g.score for g in compatible])[::-1]
        candidates = [compatible[compatible_indices[i]] for i in range(min(len(compatible_indices), count))]

        if len(candidates) < count:
            uncompatible_indices = np.argsort([g.score for g in uncompatible])[::-1]
            candidates.extend([uncompatible[uncompatible_indices[i]] for i in range(min(len(uncompatible_indices), count - len(candidates)))])

        for c in candidates:
            self._archive.remove(c)

        return candidates

    def get_size(self) -> int:
        return len(self._archive)
