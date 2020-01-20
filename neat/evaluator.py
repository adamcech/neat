import math
import time
from typing import List, Any

from dataset.dataset import Dataset
from neat.ann.ann import Ann
from neat.encoding.genotype import Genotype

import multiprocessing as mp
import numpy as np


class Evaluator:

    def __init__(self, population: List[Genotype], dataset: Dataset, forced: bool, seed: Any = None, **kwargs):
        self.population = population
        self.dataset = dataset
        self.forced = forced
        self.seed = seed

        self.step = kwargs.get("cpu_step", None)
        self.step = 5 if self.step is None else self.step

    def test_avg_score(self) -> float:
        all_scores = []

        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []

        seeds_count = len(self.seed)
        step = math.ceil(seeds_count / process_count)
        step = 5 if step > 5 else step

        # noinspection PyBroadException
        try:
            for genotype in self.population:
                f = -step
                while f + step < seeds_count:
                    running = sum([1 for p in process_list if p.is_alive()])
                    if running < process_count:
                        f += step
                        t = f + step
                        if t > seeds_count:
                            t = seeds_count

                        process_out = mp.Queue()
                        process_out_list.append(process_out)

                        p = mp.Process(target=Evaluator.__find_best_seed_single, args=(genotype, self.dataset, self.seed[f:t], process_out))
                        p.start()
                        process_list.append(p)

                    time.sleep(0.01)

                for p in process_list:
                    p.join()

            # save results
            for out_list in process_out_list:
                avg_score, envs, scores = out_list.get()
                all_scores.extend(scores)
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                avg_score, seeds, scores = self.dataset.get_fitness(Ann(genotype), self.seed)
                all_scores.extend(scores)

        all_scores = [s for s in all_scores if s > -25]
        return sum(all_scores) / len(all_scores)

    def find_best_seed(self, count: int) -> List[Any]:
        seeds_scores = [-999999.99 for _ in range(len(self.seed))]
        seeds_scores_envs_indices = {self.seed[i]: i for i in range(len(self.seed))}

        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []

        seeds_count = len(self.seed)
        step = math.ceil(seeds_count / process_count)
        step = 5 if step > 5 else step

        # noinspection PyBroadException
        try:
            for genotype in self.population:
                f = -step
                while f + step < seeds_count:
                    running = sum([1 for p in process_list if p.is_alive()])
                    if running < process_count:
                        f += step
                        t = f + step
                        if t > seeds_count:
                            t = seeds_count

                        process_out = mp.Queue()
                        process_out_list.append(process_out)

                        p = mp.Process(target=Evaluator.__find_best_seed_single, args=(genotype, self.dataset, self.seed[f:t], process_out))
                        p.start()
                        process_list.append(p)

                    time.sleep(0.01)

                for p in process_list:
                    p.join()

            # save results
            for out_list in process_out_list:
                avg_score, envs, scores = out_list.get()
                for i in range(len(envs)):
                    lst_index = seeds_scores_envs_indices[envs[i]]
                    if scores[i] > seeds_scores[lst_index]:
                        seeds_scores[lst_index] = scores[i]
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                avg_score, seeds, scores = self.dataset.get_fitness(Ann(genotype), self.seed)

                for i in range(len(seeds_scores)):
                    if scores[i] > seeds_scores[i]:
                        seeds_scores[i] = scores[i]

        seeds_indices = np.argsort(seeds_scores)[::-1]
        for i in range(min(100, len(seeds_scores))):
            print(seeds_scores[seeds_indices[i]], self.seed[seeds_indices[i]])

        return [self.seed[seeds_indices[i]] for i in range(count)]

    @staticmethod
    def __find_best_seed_single(genotype: Genotype, dataset: Dataset, seed: List[Any], process_out: mp.Queue):
        process_out.put(dataset.get_fitness(Ann(genotype), seed))

    def calculate_score(self):
        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []
        process_seed_list = []
        process_scores_list = []
        pop_size = len(self.population)

        # noinspection PyBroadException
        try:
            f = -self.step
            while f < pop_size:
                running = sum([1 for p in process_list if p.is_alive()])
                if running < process_count:
                    f += self.step
                    t = f + self.step
                    if t > pop_size:
                        t = pop_size

                    process_out = mp.Queue()
                    process_out_list.append(process_out)

                    process_seed = mp.Queue()
                    process_seed_list.append(process_seed)

                    process_scores = mp.Queue()
                    process_scores_list.append(process_scores)

                    p = mp.Process(target=Evaluator.__calculate_score_single,
                                   args=(self.population[f:t], self.dataset, process_out, process_seed, process_scores,
                                         self.forced, self.seed))
                    p.start()
                    process_list.append(p)

                time.sleep(0.01)

            for p in process_list:
                p.join()

            # save results
            f = -self.step
            for i in range(len(process_out_list)):
                out_list = process_out_list[i]
                seed_list = process_seed_list[i]
                scores_list = process_scores_list[i]
                f += self.step
                t = f + self.step
                if t > pop_size:
                    t = pop_size

                for j in range(f, t):
                    self.population[j].evaluated_seed = seed_list.get()
                    self.population[j].fitness = out_list.get()
                    self.population[j].score = self.population[j].fitness
                    self.population[j].scores = scores_list.get()
                    self.population[j].evaluated = True
                    self.population[j].evaluated_fitness = self.population[j].fitness
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in self.population:
                genotype.calculate_fitness(self.dataset, self.forced, self.seed)

    @staticmethod
    def __calculate_score_single(population: List[Genotype], dataset: Dataset, process_out: mp.Queue, process_seed: mp.Queue,
                                 process_scores: mp.Queue, forced: bool = False, seed: Any = None):
        for genotype in population:
            genotype.calculate_fitness(dataset, forced, seed)
            process_out.put(genotype.fitness)
            process_seed.put(genotype.evaluated_seed)
            process_scores.put(genotype.scores)
