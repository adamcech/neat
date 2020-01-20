import math
import random
import multiprocessing as mp
import time

import numpy as np

from typing import Any, List, Tuple

from dataset.dataset import Dataset
from neat.config import Config
from neat.encoding.edge import Edge
from neat.encoding.genotype import Genotype
from neat.encoding.node import Node


class GenotypeTrainer:

    def __init__(self, dataset: Dataset, population: List[Genotype], config: Config):
        self._dataset = dataset

        self._original_population = population
        self._size = len(self._original_population)

        self._max_iterations = config.train_max_iterations
        self._elitism = math.ceil(self._size * config.train_elitism)
        self._f = config.train_f
        self._cr = config.train_cr

        self._best_seed = []
        self._population = self._create_population()

    def _create_population(self) -> List[Genotype]:
        """
        original_population_copy = [g.deepcopy() for g in self._original_population]
        GenotypeTrainer.__evaluate(original_population_copy, self._dataset, True, GenotypeTrainer.__generate_seed(5))
        """

        GenotypeTrainer.__evaluate(self._original_population, self._dataset, True, GenotypeTrainer.__generate_seed(5))

        original_population_sort_indices = np.argsort([p.score for p in self._original_population])
        best = self._original_population[original_population_sort_indices[-1]]
        print(best)

        seed = best.evaluated_seed
        seed_result = []

        for s in seed:
            best.calculate_fitness(self._dataset, True, [s])
            seed_result.append(best.score)

        self._best_seed = [seed[seed_result.index(max(seed_result))]]

        best_nodes = [(node.id, node.type) for node in best.nodes]
        best_edges = [(edge.innovation, edge.input, edge.output, edge.enabled) for edge in best.edges if edge.enabled]

        population = []

        for i in range(self._size):
            genotype = Genotype()
            genotype.inputs = best.inputs
            curr_edges = {edge.innovation: edge.weight for edge in self._original_population[i].edges}

            for node_id, node_type in best_nodes:
                genotype.nodes.append(Node(node_id, node_type))

            for edge_innovation, edge_input, edge_output, edge_enabled in best_edges:
                genotype.edges.append(Edge(edge_input, edge_output, edge_enabled, edge_innovation, weight=curr_edges.get(edge_innovation), mutable=False))

            population.append(genotype)

        return population

    def train(self):
        seed = GenotypeTrainer.__generate_seed(100)
        start_seed = -1

        for generation in range(self._max_iterations):
            if generation % 10 == 0:
                start_seed += 1
                print(self._best_seed + seed[start_seed:start_seed+1])
                GenotypeTrainer.__evaluate(self._population, self._dataset, True, self._best_seed + seed[start_seed:start_seed+1])

            sort_indices = np.argsort([p.fitness for p in self._population])[::-1]
            elite_pop = [self._population[sort_indices[i]] for i in range(self._elitism)]
            offsprings = []

            for i in range(self._size):
                parents = random.sample(self._population, 2)  # type: List[Genotype]
                best = random.sample(elite_pop, 1)[0]  # type: Genotype
                offspring = Genotype()
                offspring.inputs = best.inputs

                for node in best.nodes:
                    offspring.nodes.append(Node(node.id, node.type))

                for j in range(len(best.edges)):
                    edge = self._population[i].edges[j]

                    p0 = parents[0].edges[j].weight
                    p1 = parents[1].edges[j].weight
                    pb = best.edges[j].weight
                    c = edge.weight
                    weight = c + (pb - c) * self._f + (p0 - p1) * self._f if np.random.uniform() < self._cr and edge.enabled else c

                    offspring.edges.append(Edge(edge.input, edge.output, edge.enabled, edge.innovation, weight=weight, mutable=False))

                offsprings.append(offspring)

            # Replace better offsprings
            GenotypeTrainer.__evaluate(offsprings, self._dataset, True, self._best_seed + seed[start_seed:start_seed+1])

            replaced = 0
            for i in range(self._size):
                if offsprings[i].score > self._population[i].score:
                    replaced += 1
                    self._population[i] = offsprings[i]

            print(generation, replaced, GenotypeTrainer.__get_stats(self._population))

    def merge(self) -> Tuple[List[Genotype], List[Any]]:
        """
        seed = GenotypeTrainer.__generate_seed(5)
        new_pop = self._population + self._original_population
        self.__evaluate(new_pop, self._dataset, True, seed)
        sort_indices = np.argsort([p.score for p in new_pop])[::-1]
        return [new_pop[sort_indices[i]] for i in range(self._size)
        """

        """
        for genotype in self._population:
            best_score = max(genotype.scores)
            best_seed = genotype.evaluated_seed[genotype.scores.index(best_score)]

            genotype.evaluated = True
            genotype.evaluated_fitness = best_score
            genotype.score = best_score
            genotype.fitness = best_score
            genotype.evaluated_seed = [best_seed]
            genotype.scores = [best_score]
        """

        sort_indices = np.argsort([p.score for p in self._population])[::-1]
        best = self._population[sort_indices[0]]

        seed = best.evaluated_seed
        seed_result = []

        for s in seed:
            best.calculate_fitness(self._dataset, True, [s])
            seed_result.append(best.score)

        best_seed = [seed[seed_result.index(max(seed_result))]]
        return self._population, best_seed

    @staticmethod
    def __get_stats(population: List[Genotype]):
        scores = [p.score for p in population]
        return max(scores), sum(scores) / len(scores)

    @staticmethod
    def __generate_seed(k: int):
        return random.sample(range(1000000), k)

    @staticmethod
    def __evaluate(population: List[Genotype], dataset: Dataset, forced: bool = False, seed: Any = None):
        process_count = mp.cpu_count()
        process_list = []  # type: List[mp.Process]
        process_out_list = []
        process_seed_list = []
        process_scores_list = []
        pop_size = len(population)

        step = 4

        # noinspection PyBroadException
        try:
            f = -step
            while f < pop_size:
                running = sum([1 for p in process_list if p.is_alive()])
                if running < process_count:
                    f += step
                    t = f + step
                    if t > pop_size:
                        t = pop_size

                    process_out = mp.Queue()
                    process_out_list.append(process_out)

                    process_seed = mp.Queue()
                    process_seed_list.append(process_seed)

                    process_scores = mp.Queue()
                    process_scores_list.append(process_scores)

                    p = mp.Process(target=GenotypeTrainer.__eval_single, args=(
                        population[f:t], dataset, process_out, process_seed, process_scores, forced, seed))
                    p.start()
                    process_list.append(p)

                time.sleep(0.01)

            for p in process_list:
                p.join()

            # save results
            f = -step
            for i in range(len(process_out_list)):
                out_list = process_out_list[i]
                seed_list = process_seed_list[i]
                scores_list = process_scores_list[i]
                f += step
                t = f + step
                if t > pop_size:
                    t = pop_size

                for j in range(f, t):
                    population[j].evaluated_seed = seed_list.get()
                    population[j].fitness = out_list.get()
                    population[j].scores = scores_list.get()
                    population[j].score = population[j].fitness
                    population[j].evaluated = True
                    population[j].evaluated_fitness = population[j].fitness
        except Exception:
            print("Multi process evaluation failed, using single process!")
            for genotype in population:
                genotype.calculate_fitness(dataset, forced, seed)

        m = min([genotype.fitness for genotype in population])
        m = abs(m) if m < 0 else 0

        for genotype in population:
            genotype.fitness += m

    @staticmethod
    def __eval_single(sub_population: List[Genotype], dataset: Dataset, process_out: mp.Queue, process_seed: mp.Queue, process_scores: mp.Queue,
                      forced: bool = False, seed: Any = None):
        for genotype in sub_population:
            genotype.calculate_fitness(dataset, forced, seed)
            process_out.put(genotype.fitness)
            process_seed.put(genotype.evaluated_seed)
            process_scores.put(genotype.scores)
