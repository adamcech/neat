import pickle
import time
from typing import List

from neat.population import Population

from neat.observers.abstract_observer import AbstractObserver
from neat.observers.autosave_observer import AutosaveObserver

from neat.config import Config


class Neat:

    def __init__(self, config: Config):
        self.config = config
        self.population = Population(self.config)
        self.observers = []  # type: List[AbstractObserver]

    def run(self):
        while self.config.generation < self.config.max_generations and self.config.evals < self.config.max_evals and \
                (self.population.get_best_member().score <= self.config.max_score_termination if self.config.max_score_termination is not None else True):
            self._notify_start_generation()
            self._next_generation()
            self.config.generation += 1
            self._notify_end_generation()

    def _next_generation(self) -> None:
        if self.config.seed_next is not None and self.config.seed_max is not None and \
           self.config.seed_attempts is not None and self.config.seed_select is not None:

            if self.config.generation % self.config.seed_next == 0 and self.config.generation != 0:
                self.population.next_seed()

        self.population.speciate()
        self.population.archivate()

        self.population.crossover()
        self.population.mutate()

        self.population.evaluate_offsprings()
        self.population.merge()

    def add_observer(self, observer: AbstractObserver) -> None:
        self.observers.append(observer)

        # Autosave observer as last, otherwise changes in other observers wont be saved
        new_observers = [observer for observer in self.observers if type(observer) == AutosaveObserver]
        for observer in [observer for observer in self.observers if type(observer) != AutosaveObserver]:
            new_observers.append(observer)

        self.observers = new_observers

    def _notify_start_generation(self):
        for observer in self.observers:
            observer.start_generation(self.config.generation)

    def _notify_end_generation(self):
        for observer in self.observers:
            observer.end_generation(self)

    @staticmethod
    def open(path: str) -> "Neat":
        with open(path, 'rb') as file:
            neat = pickle.load(file)
        return neat
