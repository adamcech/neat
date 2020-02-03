import time
from typing import List, Any

from neat.ann.ann import Ann
from neat.config import Config
from neat.encoding.genotype import Genotype

import multiprocessing as mp
import numpy as np


class Evaluator:

    def __init__(self, population: List[Genotype], config: Config, forced: bool, seed: Any = None):
        self.population = population
        self.config = config
        self.forced = forced
        self.seed = seed

    def test_avg_score(self) -> float:
        all_scores = []

        process_list = []  # type: List[mp.Process]
        process_out_list = []

        seeds_count = len(self.seed)

        # noinspection PyBroadException
        try:
            for genotype in self.population:
                f = -self.config.mp_step
                while f + self.config.mp_step < seeds_count:
                    running = sum([1 for p in process_list if p.is_alive()])
                    if running < self.config.mp_max_proc:
                        f += self.config.mp_step
                        t = f + self.config.mp_step
                        if t > seeds_count:
                            t = seeds_count

                        process_out = mp.Queue()
                        process_out_list.append(process_out)

                        p = mp.Process(target=Evaluator.__find_best_seed_single, args=(genotype, self.config, self.seed[f:t], process_out))
                        p.start()
                        process_list.append(p)

                    time.sleep(0.01)

                for p in process_list:
                    p.join()

            # save results
            for out_list in process_out_list:
                avg_score, envs, scores, evals = out_list.get()
                self.config.evals += evals
                all_scores.extend(scores)
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                avg_score, seeds, scores, evals = self.config.dataset.get_fitness(Ann(genotype, self.config.activation), self.seed)
                self.config.evals += evals
                all_scores.extend(scores)

        all_scores = [s for s in all_scores if s > -25]
        return sum(all_scores) / len(all_scores)

    def find_best_seed(self, count: int) -> List[Any]:
        seeds_scores = [-999999.99 for _ in range(len(self.seed))]
        seeds_scores_envs_indices = {self.seed[i]: i for i in range(len(self.seed))}

        process_list = []  # type: List[mp.Process]
        process_out_list = []

        seeds_count = len(self.seed)

        # noinspection PyBroadException
        try:
            for genotype in self.population:
                f = -self.config.mp_step
                while f + self.config.mp_step < seeds_count:
                    running = sum([1 for p in process_list if p.is_alive()])
                    if running < self.config.mp_max_proc:
                        f += self.config.mp_step
                        t = f + self.config.mp_step
                        if t > seeds_count:
                            t = seeds_count

                        process_out = mp.Queue()
                        process_out_list.append(process_out)

                        p = mp.Process(target=Evaluator.__find_best_seed_single, args=(genotype, self.config, self.seed[f:t], process_out))
                        p.start()
                        process_list.append(p)

                    time.sleep(0.01)

                for p in process_list:
                    p.join()

            # save results
            for out_list in process_out_list:
                avg_score, envs, scores, evals = out_list.get()
                self.config.evals += evals
                for i in range(len(envs)):
                    lst_index = seeds_scores_envs_indices[envs[i]]
                    if scores[i] > seeds_scores[lst_index]:
                        seeds_scores[lst_index] = scores[i]
        except Exception as e:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                avg_score, seeds, scores, evals = self.config.dataset.get_fitness(Ann(genotype, self.config.activation), self.seed)
                self.config.evals += evals

                for i in range(len(seeds_scores)):
                    if scores[i] > seeds_scores[i]:
                        seeds_scores[i] = scores[i]

        seeds_indices = np.argsort(seeds_scores)[::-1]
        return [self.seed[seeds_indices[i]] for i in range(count)]

    @staticmethod
    def __find_best_seed_single(genotype: Genotype, config: Config, seed: List[Any], process_out: mp.Queue):
        process_out.put(config.dataset.get_fitness(Ann(genotype, config.activation), seed))

    def calculate_score(self):
        process_list = []  # type: List[mp.Process]
        process_out_list = []
        pop_size = len(self.population)

        # noinspection PyBroadException
        try:
            f = -self.config.mp_step
            while f + self.config.mp_step < pop_size:
                running = sum([1 for p in process_list if p.is_alive()])
                if running < self.config.mp_max_proc:
                    f += self.config.mp_step
                    t = f + self.config.mp_step
                    if t > pop_size:
                        t = pop_size

                    process_out = mp.Queue()
                    process_out_list.append(process_out)

                    p = mp.Process(target=Evaluator.__calculate_score_single,
                                   args=(self.population[f:t], self.config, process_out, self.forced, self.seed))
                    p.start()
                    process_list.append(p)

                time.sleep(0.01)

            for p in process_list:
                p.join()

            # save results
            f = -self.config.mp_step
            for out_list in process_out_list:
                f += self.config.mp_step
                t = f + self.config.mp_step
                if t > pop_size:
                    t = pop_size

                for j in range(f, t):
                    g = self.population[j]
                    g.evaluated = True
                    g.score, g.evaluated_seed, g.scores, evals = out_list.get()
                    self.config.evals += evals
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                score, evaluated_seed, scores, evals = genotype.calculate_fitness(self.config, self.forced, self.seed)
                self.config.evals += evals

    @staticmethod
    def __calculate_score_single(population: List[Genotype], config: Config, process_out: mp.Queue, forced: bool = False, seed: Any = None):
        for genotype in population:
            process_out.put(genotype.calculate_fitness(config, forced, seed))
