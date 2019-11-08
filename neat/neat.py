import time
from typing import List

from dataset.dataset import Dataset
from neat.population import Population
from neat.observers.abstract_observer import AbstractObserver


class Neat:
    """Neuroevolution of augmenting topologies

    Args:
         c1 (float): Excess Genes Importance
         c2 (float): Disjoint Genes Importance
         c3 (float): Weights difference Importance
         t (float): Compatibility Threshold
         population_size (int): Population Size
         dataset (Dataset): Dataset to use
    """

    def __init__(self, c1: float, c2: float, c3: float, t: float, population_size: int, dataset: Dataset):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.t = t

        self.dataset = dataset

        self.population_size = population_size
        self._population = Population(self)

        self.generation = 0

        self.observers = []  # type: List[AbstractObserver]

    def next_generations(self, generations: int) -> None:
        max_generation = self.generation + generations
        while self.generation < max_generation:
            self._notify_start_generation()
            self._next_generation()
            self._notify_end_generation()
            self.generation += 1

    def _next_generation(self) -> None:
        self._population.speciate()
        self._population.crossover()

        self._population.mutate_weights()
        self._population.mutate_topology()

        # self._population.mutate_add_edge()
        # self._population.mutate_add_node()

        self._population.evaluate(self.dataset)

    def add_observer(self, reporter: AbstractObserver) -> None:
        self.observers.append(reporter)

    def _notify_start_generation(self):
        for observer in self.observers:
            observer.start_generation(self.generation)

    def _notify_end_generation(self):
        for observer in self.observers:
            observer.end_generation(self._population)
