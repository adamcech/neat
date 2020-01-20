import pickle
import time
from typing import List

from dataset.dataset import Dataset
from neat.observers.autosave_observer import AutosaveObserver
from neat.population import Population
from neat.observers.abstract_observer import AbstractObserver

from neat.config import Config


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

    def __init__(self, config: Config, dataset: Dataset):
        self.config = config
        self.dataset = dataset

        self.population_size = config.population_size
        self.population = Population(self.config, self.dataset)

        self.generation = 0
        self.seed_postpone = 0

        self.observers = []  # type: List[AbstractObserver]

    def next_generations(self, generations: int) -> None:
        max_generation = self.generation + generations
        while self.generation < max_generation:
            self._notify_start_generation()
            self._next_generation()
            self.generation += 1
            self._notify_end_generation()

    def _next_generation(self) -> None:
        """
        self.population.speciate()
        self.population.crossover()
        self.population.mutate_weights()
        self.population.mutate_topology()

        self.population.evaluate()


        if self.generation % self.train_counter == 0 and self.generation != 0:
            self.population.train_nets()
        """
        # next_seed = self.generation % min(50, int(25 + self.generation / 50)) == 0
        next_seed = self.generation % (25 if self.generation < 400 else 50) == 0
        if next_seed and self.seed_postpone <= 0 and self.generation != 0:
            self.population.next_seed(1 if self.generation < 400 else 2)
            self.population.new_evaluate_ancestors()
        self.seed_postpone -= 1

        self.population.speciate()
        self.population.new_boost_species()

        self.population.new_crossover()
        self.population.new_mutations()

        self.population.new_evaluate()
        self.population.merge()
        # self.population.new_mutate_topology_forced()

        self.population.save_stats()

        """
        if self.population.is_stagnating():
            self.population.train_nets()
            self.seed_postpone = 10
        """

    def add_observer(self, observer: AbstractObserver) -> None:
        self.observers.append(observer)

        new_observers = [observer for observer in self.observers if type(observer) == AutosaveObserver]
        for observer in [observer for observer in self.observers if type(observer) != AutosaveObserver]:
            new_observers.append(observer)

        self.observers = new_observers

    def _notify_start_generation(self):
        for observer in self.observers:
            observer.start_generation(self.generation)

    def _notify_end_generation(self):
        for observer in self.observers:
            observer.end_generation(self)

    @staticmethod
    def open(path: str) -> "Neat":
        with open(path, 'rb') as file:
            neat = pickle.load(file)
        return neat
