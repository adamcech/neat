from typing import List

from neat.encoding.edge import Edge


class WeightMap:

    def __init__(self):
        self._map = {}

    @staticmethod
    def init_from_population(population: List["Genotype"]) -> "WeightMap":
        weight_map = WeightMap()

        for genotype in population:
            for edge in genotype.edges:
                weight_map._add_edge(edge)

        return weight_map

    def _add_edge(self, edge: Edge):
        edge_list = self._map.get(edge.innovation)

        if edge_list is None:
            self._map[edge.innovation] = []

        if edge.weight not in self._map[edge.innovation]:
            self._map[edge.innovation].append(edge.weight)

    def get_edge_weights(self, edge_innovation: int) -> List[float]:
        return self._map.get(edge_innovation)
