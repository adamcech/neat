import os
from typing import List, Union, Any

from neat.config import Config
from neat.observers.abstract_observer import AbstractObserver
from neat.strategies.population_strategies import PopulationStrategies


class CsvObserver(AbstractObserver):

    def __init__(self, config: Config):
        self.config = config

        self.other_csv = self.config.work_dir + "other.csv"
        self.nodes_csv = self.config.work_dir + "nodes.csv"
        self.edges_csv = self.config.work_dir + "edges.csv"
        self.score_csv = self.config.work_dir + "score.csv"
        self.species_csv = self.config.work_dir + "species.csv"

        other_header = ["generation"]

        self._print(self.other_csv, self._convert_list_to_csv(other_header))

    def start_generation(self, generation: int) -> None:
        pass

    def _convert_list_to_csv(self, lst: List[Any]) -> str:
        s = str(self.config.generation) + ";"
        for col in lst:
            s += str(col) + ";"
        return s

    def end_generation(self, neat: "Neat") -> None:
        other = []

        nodes = [sum(1 for n in genotype.nodes if n.is_hidden() or n.is_output()) for genotype in neat.population.population]
        edges = [sum(1 for e in genotype.edges if e.enabled) for genotype in neat.population.population]
        scores = [g.score for g in neat.population.population]
        species = [s.score for s in neat.population.species]

        self._print(self.other_csv, other)
        self._print(self.nodes_csv, nodes)
        self._print(self.edges_csv, edges)
        self._print(self.score_csv, scores)
        self._print(self.species_csv, species)

    def _print(self, file_path: str, s: Union[str, List[Any]]):
        if type(s) is list:
            s = self._convert_list_to_csv(s)

        if not os.path.exists(self.config.work_dir):
            os.makedirs(self.config.work_dir)

        if not os.path.exists(file_path):
            open(file_path, "a").close()

        file = open(file_path, "a")
        file.write(s + "\n")
        file.close()
