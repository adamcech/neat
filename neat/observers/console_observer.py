import time
from typing import List

import numpy

from neat.population import Population
from neat.observers.abstract_observer import AbstractObserver


class ConsoleObserver(AbstractObserver):

    def __init__(self, stats=True, species=True):
        self._stats = stats
        self._species = species

        self._generation = 0
        self._generation_start_time = 0  # type: float
        self._eval_times = []  # type: List[float]

        self._max_col_length = 16

        self._cols = ["Generation", "Avg. Eval.", "Eval. Time", "Grow Rate", "Compatibility", "Best Score", "Avg. Score", "Species [id, members, fitness]"]
        self._print_cols(self._cols)

    def start_generation(self, generation: int) -> None:
        self._generation = generation
        self._generation_start_time = time.time()

    def end_generation(self, population: Population) -> None:
        eval_time = time.time() - self._generation_start_time

        if len(self._eval_times) == 0 or eval_time < self._get_avg_eval() * 10:
            self._eval_times.append(eval_time)

        cols = []

        if self._stats:
            cols.append(self._generation)
            cols.append(self._get_avg_eval())
            cols.append(eval_time)
            cols.append(population.get_grow_rate())
            cols.append(population.get_compatibility())
            cols.append(population.get_best().score)
            cols.append(population.get_avg_score())

        species = []
        if self._species:
            for s in population.get_species():
                species.append([s.id, len(s.members), round(s.score, 3)])

        cols.append(str(len(population.get_species())) + " " + str(species))
        self._print_cols(cols)

    def _get_avg_eval(self) -> float:
        return sum(self._eval_times) / len(self._eval_times)

    def _print_cols(self, cols: list):
        cols = [round(col, 3) if type(col) == float or numpy.isreal(col) else col for col in cols]
        cols = [str(col) for col in cols]

        for col in cols:
            print(col.ljust(self._max_col_length), end="")
        print()
