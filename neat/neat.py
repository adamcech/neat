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

        self.best_genotype = None
        self.best_genotype_score = None

        self.best_genotypes = []

        self.min_score_history = []
        self.avg_score_history = []
        self.max_score_history = []
        self.max_score_history_gens = []

    def start(self):
        print(self.config.generation < self.config.max_generations)
        print(self.config.evals < self.config.max_evals)
        print(not self.config.done)
        print("run", self.config.generation < self.config.max_generations and self.config.evals < self.config.max_evals and not self.config.done, end="; ")

        while self.config.generation < self.config.max_generations and self.config.evals < self.config.max_evals and not self.config.done:
            print("start notify", end="; ")
            self._notify_start_generation()
            print("next generation", end="; ")
            self._next_generation()
            print("generation done", end="; ")
            self.config.generation += 1
            self._notify_end_generation()
            print("end notify", end="; ")
            print()

    def _next_generation(self) -> None:
        print("gen started", end="; ")

        if self.config.generation % self.config.seed_next == 0 and self.config.generation > 0:
            print("seed", end="; ")
            print("SEED OLD", self.population.seed)
            self.population.next_seed()
            print("SEED NEW", self.population.seed)

        if self.config.generation % 50 == 0 and self.config.generation > 0:
            min_score, avg_score, max_score, current_best_genotype = self.population.test()
            self.min_score_history.append(min_score)
            self.avg_score_history.append(avg_score)
            self.max_score_history.append(max_score)
            self.max_score_history_gens.append(self.config.generation)

            if self.best_genotype is None or self.best_genotype_score < max_score:
                self.best_genotype = current_best_genotype.copy()
                self.best_genotype_score = max_score
                self.best_genotypes.append(self.best_genotype)

            if max_score >= self.config.max_score_termination:
                print("SOLVED ENDING...")
                self.config.done = True
                self.config.done_genotype = current_best_genotype.copy()
                self.config.done_genotype.score = max_score

        self.population.adapt_params_start()

        print("speciate", end="; ")
        start = time.time()
        self.population.speciate()
        end = time.time()
        print("TIME speciate end", end - start)

        print("archivate", end="; ")
        start = time.time()
        self.population.archivate()
        end = time.time()
        print("TIME archivate end", end - start)

        # print("crossover", end="; ")
        # start = time.time()
        # self.population.crossover_mutate_ray()
        # end = time.time()
        # print("\n\nTIME crossover_mutate_ray end", end - start)

        print("crossover", end="; ")
        start = time.time()
        self.population.crossover()
        end = time.time()
        print("TIME crossover end", end - start)

        print("mutate topology", end="; ")
        start = time.time()
        self.population.mutate_topology()
        end = time.time()
        print("TIME mutate end", end - start)

        print("evaluate offspring", end="; ")
        start = time.time()
        self.population.evaluate_offsprings()
        end = time.time()
        print("TIME EVAL OFFSPRINGS end", end - start)

        print("merge", end="; ")
        start = time.time()
        self.population.merge()
        end = time.time()
        print("TIME MERGE end", end - start)

        # self.population.mutate_weights()

        print("adapt params", end="; ")
        self.population.adapt_params_end()

        print("BEST DISABLED EDGES", sum(1 for e in self.population.get_best_member().edges if not e.enabled))

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
        print("Openning: " + path)

        with open(path, 'rb') as file:
            neat = pickle.load(file)  # type: Neat

        neat.config.next_tcp_port()

        for obs in neat.observers:
            if type(obs) == AutosaveObserver:
                obs.init_walltime(neat.config.walltime_sec)

        return neat
