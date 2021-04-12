import time
import os
from typing import List, Tuple

import numpy as np

from neat.config import Config
from neat.observers.abstract_observer import AbstractObserver
from neat.strategies.population_strategies import PopulationStrategies


class ConsoleObserver(AbstractObserver):

    def __init__(self, redirect_path: str = None, config: Config = None, stats=True, species=True):
        self.dir_path_redirect = redirect_path

        self._stats = stats
        self._species = species

        self._generation = 0
        self._generation_start_time = 0  # type: float
        self._eval_times = []  # type: List[float]

        self._max_col_length = 16

        self._cols = ["Generation", "Avg. Eval.", "Before", "After", "Arch. Len", "Cmp   Stg", "Evals", "Best Score", "Avg. Score", "Edges", "Nodes", "Seed Avg", "Strategy"]

        if config is not None:
            self._print_redirect(str(config))

        self._print_cols(self._cols)

    def start_generation(self, generation: int) -> None:
        self._generation = generation
        self._generation_start_time = time.time()

    def end_generation(self, neat: "Neat") -> None:
        population = neat.population

        eval_time = time.time() - self._generation_start_time

        if len(self._eval_times) == 0 or eval_time < self._get_avg_eval() * 100:
            self._eval_times.append(eval_time)

        cols = []

        if self._stats:
            cols.append(str(self._generation) + "   " + str(round(eval_time, 1)))
            cols.append(self._get_avg_eval())

            cols.append(str(round(neat.population.comp_before, 2)).ljust(5) + "  " + str(neat.population.specs_before).ljust(5))
            cols.append(str(round(neat.population.comp_after, 2)).ljust(5) + "  " + str(neat.population.specs_after).ljust(5))

            cols.append(population.archive.get_info())

            cols.append(str(round(population.get_compatibility(), 2)).ljust(7) + str(neat.population.stagnating))

            evals = str(population.config.evals)
            evals = [evals[len(evals) - 1 - i] if i == 0 or i % 3 != 0 else evals[len(evals) - 1 - i] + " " for i in range(len(evals))]
            evals_str = ""
            for i in range(len(evals) - 1, -1, -1):
                evals_str = evals_str + evals[i]
            cols.append(evals_str)

            cols.append(population.get_best_member().score)
            cols.append(sum(p.score for p in population.population) / len(population.population))

            e_max, e_avg, e_min = self._get_edges_info(neat.population)
            cols.append(str(e_max) + " " + str(e_avg) + " " + str(e_min))

            n_max, n_avg, n_min = self._get_nodes_info(neat.population)
            cols.append(str(n_max) + " " + str(n_avg) + " " + str(n_min))

            test_hist = "None"
            if len(neat.max_score_history) > 0:
                test_hist = str(int(neat.max_score_history[-1])) + " " + str(int(neat.avg_score_history[-1])) + " " + str(int(neat.min_score_history[-1]))

            cols.append(test_hist)

            strategies = neat.population.crossover_manager.strategies
            if neat.config.learning_period >= 10:
                strategy = str(strategies[(neat.config.generation - 1) % len(strategies)])

                if (neat.config.generation - 1) % (2 * len(strategies)) >= len(strategies):
                    strategy = ""
            else:

                strategy = str(strategies[(neat.config.generation - 1) % len(strategies)])

            if neat.config.generation % neat.config.learning_period == 0:
                strategy = "-----------------------------"

            cols.append(strategy)

        species = []
        if self._species:
            for s in population.species:
                species.append([s.id, len(s.members), round(s.score, 3)])

        self._print_cols(cols)

        # if neat.config.generation % 10 == 0 and neat.config.generation > 0:
        #     self._print(neat.population.config.dataset.render(Ann(neat.population.get_best_member(), neat.config.activation, neat.config.max_layers)) + "\n")
        #
        # if neat.config.done:
        #     self._print_redirect("!!! SOLVED !!! ENDING !!!")

    def _get_avg_eval(self) -> float:
        return sum(self._eval_times) / len(self._eval_times)

    def _print_cols(self, cols: list):
        cols = [round(col, 3) if type(col) == float or np.isreal(col) else col for col in cols]
        cols = [str(col) for col in cols]

        s = ""
        for col in cols:
            s += col.ljust(self._max_col_length)
        s += "\n"

        self._print(s, end="\n")

    def _print(self, s: str, **kwargs):
        print("\n" + s, end=kwargs.get("end", "\n"))

        if self.dir_path_redirect is not None:
            self._print_redirect(s)

    def _print_redirect(self, s: str):
        if not os.path.exists(self.dir_path_redirect):
            os.makedirs(self.dir_path_redirect)

        console_file = self.dir_path_redirect + "console"
        if not os.path.exists(console_file):
            open(console_file, "a").close()

        file = open(console_file, "a")
        file.write(s)
        file.close()

    def _get_nodes_info(self, population: "Population") -> Tuple[int, int, int]:
        avg = []
        maximum = 0
        minimum = 9999999

        for member in population.population:
            member_nodes = len([n for n in member.nodes if n.is_hidden() or n.is_output()])

            avg.append(member_nodes)

            if member_nodes > maximum:
                maximum = member_nodes

            if member_nodes < minimum:
                minimum = member_nodes

        return maximum, int(sum(avg) / len(avg)), minimum

    def _get_edges_info(self, population: "Population") -> Tuple[int, int, int]:
        avg = []
        maximum = 0
        minimum = 9999999

        for member in population.population:
            member_edges = len([e for e in member.edges if e.enabled])

            avg.append(member_edges)

            if member_edges > maximum:
                maximum = member_edges

            if member_edges < minimum:
                minimum = member_edges

        return maximum, int(sum(avg) / len(avg)), minimum
