import os
import time
import socket
from typing import List, Any, Tuple, Union

from evaluators.cluster.cluster_node_client import ClusterNodeClient
from evaluators.evaluator import Evaluator
from neat.config import Config
from neat.encoding.genotype import Genotype

import multiprocessing as mp


class ClusterEvaluator(Evaluator):

    def __init__(self, config: Config):
        super().__init__(config)

        own_hostname = socket.gethostname()
        hostnames = [h for h in os.listdir(self.config.cluster_nodes_loc)]

        self.nodes = [ClusterNodeClient(mp.cpu_count(), h, self.config.tcp_port) for h in hostnames if own_hostname not in h]
        self.nodes.append(ClusterNodeClient(config.cluster_main_max_load, socket.gethostname(), self.config.tcp_port))  # Server as last (smallest load maybe?)

    def _run_impl(self, population: List[Genotype], seed: Union[None, List[Any]], **kwargs) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results_size, results, requests = self._create_requests_and_results(population, seed, **kwargs)

        f = 0
        while len(results) < results_size:
            for node in self.nodes:
                if node.is_ready() and f < len(requests):
                    step = max(node.max_load, int((len(requests) - f) / (len(self.nodes) * 2)))
                    t = min(f + step, len(requests))

                    node.request(requests[f:t])
                    f = t

                if node.is_done():
                    results.extend(node.get_results())

            time.sleep(0.00001)

        return results
