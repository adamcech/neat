from typing import List, Any, Union, Dict, Tuple

from neat.config import Config
from neat.encoding.genotype import Genotype

import numpy as np


class Evaluator:

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def split(a, n):
        k, m = divmod(len(a), n)
        return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

    def _create_requests_and_results(self, population: List[Genotype], seed: Union[None, List[Any]], **kwargs) -> Tuple[int, List[Tuple[int, float, Union[None, List[Any]], int]], List[Tuple[int, Genotype, Union[None, List[Any]]]]]:
        forced = kwargs.get("forced", True)

        results = []  # type: List[Tuple[int, float, Union[None, List[Any]], int]]
        requests = []  # type: List[Tuple[int, Genotype, Union[None, List[Any]]]]

        seed_len = 1 if seed is None else len(seed)
        results_size = int(len(population) * seed_len)

        if forced:
            for i in range(len(population)):
                if seed is None:
                    requests.append((i, population[i], None))
                if type(seed) == list:
                    for s in seed:
                        requests.append((i, population[i], [s]))
        else:
            for i in range(len(population)):
                genotype = population[i]

                if seed is None:
                    results.append((i, genotype.score, None, 0))
                if type(seed) == list:
                    for s in seed:
                        if s in population[i].evaluated_seed:
                            results.append((i, genotype.scores[genotype.evaluated_seed.index(s)], [s], 0))
                        else:
                            requests.append((i, genotype, [s]))

        return results_size, results, requests

    def _run(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results = self._run_impl(population, seed, **kwargs)

        if kwargs.get("evals", True):
            self.config.evals += sum([episode_evals for _, _, _, episode_evals in results])

        return results

    def _run_impl(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        raise NotImplementedError()

    def calculate_score(self, population: List[Genotype], seed: Any = None, **kwargs) -> None:
        if len(population) == 0:
            return

        results = self._run(population, seed, **kwargs)

        population_scores = {i: [] for i in range(len(population))}  # type: Dict[int, List[float]]
        population_seed = {i: [] for i in range(len(population))}  # type: Dict[int, Union[None, List[Any]]]

        # save results
        if self.config.dataset.get_seed_type() is list:
            for result in results:
                genotype_id, score, seed, evals = result
                population_seed[genotype_id].append(seed[0])
                population_scores[genotype_id].append(score)
        elif self.config.dataset.get_seed_type() is None:
            for result in results:
                genotype_id, score, seed, evals = result
                population_seed[genotype_id] = None
                population_scores[genotype_id].append(score)

        for i in range(len(population)):
            g = population[i]
            g.evaluated_seed, g.scores = population_seed[i], population_scores[i]
            g.score = sum(g.scores) / len(g.scores)

    def test(self, population: List[Genotype], seed: Any = None, **kwargs) -> Tuple[List[List[float]], List[float], List[Union[List[int], None]]]:
        if len(population) == 0 or (seed is not None and len(seed) == 0):
            raise Exception("Empty population or seed, cant test.")

        genotype_results = [[] for _ in range(len(population))]
        genotype_seeds = [[] for _ in range(len(population))]

        if seed is None:
            # TODO
            genotype_scores = [g.score for g in population]
            sort_indices = np.argsort(genotype_scores)[::-1]
            genotype_scores = [genotype_scores[i] for i in sort_indices]

            genotype_results = [[score] for score in genotype_scores]
            genotype_seeds = [None for _ in range(len(population))]

            return genotype_results, genotype_scores, genotype_seeds
        else:

            results = self._run(population, seed, **kwargs)
            scores = []

            for genotype_id, score, seed, evals in results:
                scores.append(score)
                genotype_results[genotype_id].append(score)
                genotype_seeds[genotype_id].extend(seed)

            for i in range(len(genotype_results)):
                sort_indices = np.argsort(genotype_results[i])[::-1]
                genotype_seeds[i] = [genotype_seeds[i][j] for j in sort_indices]
                genotype_results[i] = [genotype_results[i][j] for j in sort_indices]

            genotypes_scores = [sum(genotype_results[i]) / len(genotype_results[i]) for i in range(len(genotype_results))]
            return genotype_results, genotypes_scores, genotype_seeds

    def calculate_avg_score(self, population: List[Genotype], seed: Any = None, **kwargs) -> List[float]:
        if len(population) == 0:
            raise Exception("Empty population, cant eval.")

        results = self._run(population, seed, **kwargs)

        # Check if solved
        population_scores = {i: [] for i in range(len(population))}  # type: Dict[int, List[float]]

        # Save results
        for result in results:
            genotype_id, score, seed, evals = result
            population_scores[genotype_id].append(score)

        scores = [sum(population_scores[i]) / len(population_scores[i]) for i in range(len(population))]

        return scores

    def calculate_best_seed(self, count: int, population: List[Genotype], seed: Any = None, **kwargs):
        if seed is None:
            return None, None

        if len(population) == 0:
            raise Exception("Empty population, cant eval.")

        results = self._run(population, seed, **kwargs)

        seeds_scores = [-999999.99 for _ in range(len(seed))]
        seeds_scores_envs_indices = {seed[i]: i for i in range(len(seed))}

        scores = []
        for result in results:
            _, score, s, evals = result
            scores.append(score)
            index = seeds_scores_envs_indices[s[0]]
            if score > seeds_scores[index]:
                seeds_scores[index] = score

        seeds_indices = np.argsort(seeds_scores)[::-1]

        s = [s for s in scores if s > - 20]
        print(s)
        print(sum(s) / len(s))

        return [seed[seeds_indices[i]] for i in range(count)], sum(scores) / len(scores)

    def calculate_best_avg_seed(self, count: int, population: List[Genotype], seed: Any = None, **kwargs) -> Tuple[Union[None, List[Any]], float]:
        if seed is None:
            return None, sum(g.score for g in population)

        if len(seed) == 0:
            raise Exception("No seed")
        elif len(population) == 0:
            raise Exception("Empty population, cant eval.")

        results = self._run(population, seed, **kwargs)

        seeds_scores = [[] for _ in range(len(seed))]  # type: List[List[float]]
        seeds_scores_envs_indices = {seed[i]: i for i in range(len(seed))}  # type: Dict[int, int]
        scores = []

        for result in results:
            _, score, s, evals = result
            scores.append(score)
            index = seeds_scores_envs_indices[s[0]]
            seeds_scores[index].append(score)

        avg_scores = [(max(scores) + (sum(scores) / len(scores))) / 2 for scores in seeds_scores]
        seeds_indices = np.argsort(avg_scores)[::-1]
        print("--------------------------------------------------------------------------------------------------------")
        print(avg_scores)
        print(seeds_indices)
        print("--------------------------------------------------------------------------------------------------------")

        return [seed[seeds_indices[i]] for i in range(count)], sum(scores) / len(scores)

    def calculate_avg_seed(self, count: int, population: List[Genotype], seed: Any = None, **kwargs) -> Tuple[Union[None, List[Any]], float]:
        if seed is None:
            return None, sum(g.score for g in population)

        if len(seed) == 0:
            raise Exception("No seed")
        elif len(population) == 0:
            raise Exception("Empty population, cant eval.")

        results = self._run(population, seed, **kwargs)

        seeds_scores = [[] for _ in range(len(seed))]  # type: List[List[float]]
        seeds_scores_envs_indices = {seed[i]: i for i in range(len(seed))}  # type: Dict[int, int]

        for result in results:
            _, score, s, evals = result
            index = seeds_scores_envs_indices[s[0]]
            seeds_scores[index].append(score)

        avg_scores = [sum(scores) / len(scores) for scores in seeds_scores]
        seeds_indices = np.argsort(avg_scores)[::-1]
        print("--------------------------------------------------------------------------------------------------------")
        print(avg_scores)
        print(seeds_indices)
        print("--------------------------------------------------------------------------------------------------------")

        return [seed[seeds_indices[i]] for i in range(count)], sum(avg_scores) / len(avg_scores)

    def calculate_best_and_worst_seed(self, count: int, population: List[Genotype], seed: Any = None, **kwargs) -> Union[None, List[Any]]:
        if seed is None:
            return None

        if len(population) == 0:
            raise Exception("Empty population, cant eval.")

        results = self._run(population, seed, **kwargs)

        seeds_scores = [-999999.99 for _ in range(len(seed))]
        seeds_scores_envs_indices = {seed[i]: i for i in range(len(seed))}

        for result in results:
            _, score, s, evals = result
            index = seeds_scores_envs_indices[s[0]]
            if score > seeds_scores[index]:
                seeds_scores[index] = score

        seeds_indices = np.argsort(seeds_scores)[::-1]
        best_counter = 0
        worst_counter = -1
        seeds = []

        select_best = True
        while len(seeds) < count:
            if select_best:
                seeds.append(seed[seeds_indices[best_counter]])
                best_counter += 1
            else:
                seeds.append(seed[seeds_indices[worst_counter]])
                worst_counter -= 1
            select_best = not select_best

        return seeds
