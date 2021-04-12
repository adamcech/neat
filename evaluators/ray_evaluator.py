import time
from typing import List, Any, Union, Tuple

from evaluators.evaluator import Evaluator
from evaluators.results_calculator import ResultsCalculator
from neat.ann.ann import Ann
from neat.config import Config
from neat.encoding.genotype import Genotype

import ray


class RayEvaluator(Evaluator):

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config
        self.results_calculator = ResultsCalculator(self.config)

    @ray.remote
    def _ray_distribute(self, requests, **kwargs):
        genotype_id, genotype, seed = requests
        score, evaluated_seed, _, evals = self.config.dataset.get_fitness(Ann(genotype, self.config.target_function), seed, **kwargs)
        return genotype_id, score, evaluated_seed, evals

    def _run_impl(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results_size, results, requests = self._create_requests_and_results(population, seed, **kwargs)
        start = time.time()

        # if len(requests) <= 24:
        #     results.extend(self.results_calculator.calculate(requests, **kwargs))
        # else:
        #     results.extend(ray.get([self._ray_distribute.remote(self, r, **kwargs) for r in requests]))

        results.extend(ray.get([self._ray_distribute.remote(self, r, **kwargs) for r in requests]))

        print("REQUESTS", len(requests), time.time() - round(start, 3))
        return results
