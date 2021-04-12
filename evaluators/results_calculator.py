from typing import List, Tuple, Union, Any

from dataset.dataset import Dataset
from neat.ann.ann import Ann
from neat.ann.target_function import TargetFunction
from neat.config import Config

import multiprocessing as mp
import time

from neat.encoding.genotype import Genotype


class ResultsCalculator:

    def __init__(self, config: Config):
        self.config = config

    def calculate(self, request: List[Tuple[int, Genotype, Union[None, List[Any]]]], **kwargs) -> List[Tuple[int, float, Union[List[Any], None], int]]:
        process_list = []  # type: List[mp.Process]
        process_out_list = []

        f = 0
        # noinspection PyBroadException
        try:
            while f < len(request):
                running = sum([1 for p in process_list if p.is_alive()])
                if running < self.config.mp_max_proc:
                    step = max(1, int((len(request) - f) / (self.config.mp_max_proc * 2)))
                    t = min(f + step, len(request))

                    process_out = mp.Queue()
                    p = mp.Process(target=ResultsCalculator.__calculate_single, args=(request[f:t], self.config.target_function, self.config.dataset, process_out, kwargs))
                    p.start()

                    process_list.append(p)
                    process_out_list.append(process_out)
                    f = t

                time.sleep(0.0001)

            for p in process_list:
                p.join()
        except Exception as e:
            print("Multi process evaluation failed, using single process!", e)
            exit()

        results = []  # type: List[Tuple[int, float, Union[None, List[Any]], int]]

        for out_list in process_out_list:
            while not out_list.empty():
                results.append(out_list.get())
            out_list.close()

        return results

    @staticmethod
    def __calculate_single(evals: List[Tuple[int, Genotype, Union[None, List[Any]]]], activation: TargetFunction, dataset: Dataset, process_out: mp.Queue, kwargs) -> None:
        for genotype_id, genotype, seed in evals:
            score, evaluated_seed, _, evals = dataset.get_fitness(Ann(genotype, activation), seed, **kwargs)
            process_out.put((genotype_id, score, evaluated_seed, evals))
