from typing import List, Any, Union, Tuple

from evaluators.evaluator import Evaluator
from evaluators.results_calculator import ResultsCalculator
from neat.ann.ann import Ann
from neat.config import Config
from neat.encoding.genotype import Genotype


class LocalEvaluator(Evaluator):

    def __init__(self, config: Config):
        super().__init__(config)

        self.config = config
        self.results_calculator = ResultsCalculator(self.config)
        self.debug = False

        if self.config.dataset_name == "xor" or self.config.dataset_name == "iris":
            self.debug = True

    def _run_impl(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        if self.debug:
            return self.eval_single_cpu(population, seed, **kwargs)
        else:
            return self.eval_multi_cpu(population, seed, **kwargs)

    def eval_single_cpu(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results_size, results, requests = self._create_requests_and_results(population, seed, **kwargs)

        for genotype_id, genotype, seed in requests:
            score, evaluated_seed, _, evals = self.config.dataset.get_fitness(Ann(genotype, self.config.target_function), seed, **kwargs)
            results.append((genotype_id, score, evaluated_seed, evals))
        return results

    def eval_multi_cpu(self, population: List[Genotype], seed: Union[List[Any], None], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results_size, results, requests = self._create_requests_and_results(population, seed, **kwargs)
        results.extend(self.results_calculator.calculate(requests, **kwargs))
        return results
