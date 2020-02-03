import copy
import random
from typing import List, Any, Tuple, Union

from neat.ann.ann import Ann
from neat.config import Config
from neat.encoding.edge import Edge
from neat.encoding.innovation_map import InnovationMap
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType

import numpy as np


class Genotype:

    def __init__(self, ancestor: "Genotype" = None):
        self.ancestor = ancestor

        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

        self.evaluated = False
        self.fitness = 0
        self.score = 0
        self.evaluated_seed = None
        self.scores = []

    @staticmethod
    def init_population(config: Config, innovation_map: InnovationMap, hidden_layer_sizes: List[int] = None) -> List["Genotype"]:
        initial_genotype = Genotype.__init_genotype(config, innovation_map, hidden_layer_sizes)
        return [initial_genotype.__init_copy() for _ in range(config.population_size)]  # type: List[Genotype]

    def __init_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for connection in self.edges:
            connection.mutate_random_weight()
        return cp

    @staticmethod
    def __init_genotype(config: Config, innovation_map: InnovationMap, hidden_layer_sizes: List[int] = None) -> "Genotype":
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []

        genotype = Genotype()

        layer_neurons_id = Genotype.__add_layer(config, genotype, config.input_nodes + config.bias_nodes, NodeType.INPUT, innovation_map)

        for layer_size in hidden_layer_sizes:
            layer_neurons_id = Genotype.__add_layer(config, genotype, layer_size, NodeType.HIDDEN, innovation_map, layer_neurons_id)

        Genotype.__add_layer(config, genotype, config.output_nodes, NodeType.OUTPUT, innovation_map, layer_neurons_id)

        return genotype

    @staticmethod
    def __add_layer(config: Config, genotype: "Genotype", layer_size: int, layer_type: NodeType, innovation_map: InnovationMap, prev_layer_neurons_id: List[int] = None) -> List[int]:
        current_layer_neurons_id = []

        for i in range(layer_size):
            current_layer_neurons_id.append(innovation_map.next_innovation())
            genotype.nodes.append(Node(current_layer_neurons_id[-1], layer_type))

        if prev_layer_neurons_id is not None:
            for prev_id in prev_layer_neurons_id:
                for current_id in current_layer_neurons_id:
                    genotype.edges.append(Edge(config, prev_id, current_id, True, innovation_map.get_edge_innovation(prev_id, current_id)))

        return current_layer_neurons_id

    def mutate_add_node(self, config: Config, innovation_map: InnovationMap):
        if len(self.edges) > 0:
            for i in range(100):
                old_edge = random.sample(self.edges, 1)[0]  # type: Edge
                if old_edge.input in config.bias_nodes_id:
                    continue

                old_edge.enabled = False

                new_node = Node(innovation_map.get_node_innovation(old_edge.input, old_edge.output, self.nodes), NodeType.HIDDEN)
                new_edge = Edge(config, old_edge.input, new_node.id, True, innovation_map.get_edge_innovation(old_edge.input, new_node.id), weight=1)
                original_edge = Edge(config, new_node.id, old_edge.output, True, innovation_map.get_edge_innovation(new_node.id, old_edge.output), weight=old_edge.weight)

                self.nodes.append(new_node)
                self.edges.append(new_edge)
                self.edges.append(original_edge)
                break

    def mutate_add_edge(self, config: Config, innovation_map: InnovationMap):
        for try_counter in range(100):
            samples = random.sample(self.nodes, 2)
            input_node = samples[0]  # type: Node
            output_node = samples[1]  # type: Node
            if (input_node.type == NodeType.OUTPUT or output_node.type == NodeType.INPUT) or \
                    ((input_node.type == NodeType.BIAS or input_node.type == NodeType.INPUT) and (output_node.type == NodeType.BIAS or output_node.type == NodeType.INPUT)) or \
                    (input_node.type == NodeType.OUTPUT and output_node.type == NodeType.OUTPUT) or \
                    any((edge.input == input_node.id and edge.output == output_node.id) or
                        (edge.input == output_node.id and edge.output == input_node.id) for edge in self.edges):
                continue

            if not self._is_new_edge_recurrent(input_node.id, output_node.id):
                self.edges.append(Edge(config, input_node.id, output_node.id, True,
                                       innovation_map.get_edge_innovation(input_node.id, output_node.id)))
                return

        # if no edge was added, at start..
        self.mutate_add_node(config, innovation_map)

    def calculate_fitness(self, config: Config, forced: bool = False, seed: List[Any] = None) -> Tuple[float, Union[None, List[Any]], List[float], int]:
        evals = 0

        if not self.evaluated or forced:
            self.evaluated = True
            self.score, self.evaluated_seed, self.scores, evals = config.dataset.get_fitness(Ann(self, config.activation), seed)

        return self.score, self.evaluated_seed, self.scores, evals

    def _is_new_edge_recurrent(self, input_node: int, output_node: int) -> bool:
        # Search for path from output_node to input_node
        visited = []
        self._dfs(output_node, visited)
        return input_node in visited

    def _dfs(self, start_node: int, visited: List[int]):
        visited.append(start_node)

        for node in self._get_adjacent_nodes(start_node):
            if node not in visited:
                self._dfs(node, visited)

    def _get_adjacent_nodes(self, node_id: int) -> List[int]:
        nodes = []
        for edge in self.edges:
            if edge.input == node_id and edge.output not in nodes:
                nodes.append(edge.output)
        return nodes

    @staticmethod
    def crossover_triple(f: float, ancestor: "Genotype", best: "Genotype", mom: "Genotype", dad: "Genotype") -> "Genotype":
        genotype = Genotype(ancestor)

        best_edges = {edge.innovation: edge.weight for edge in best.edges}
        mom_edges = {edge.innovation: edge.weight for edge in mom.edges}
        dad_edges = {edge.innovation: edge.weight for edge in dad.edges}

        for node in ancestor.nodes:
            genotype.nodes.append(Node(node.id, node.type))

        for edge in ancestor.edges:
            c = edge.weight
            b = best_edges.get(edge.innovation)
            m = mom_edges.get(edge.innovation)
            d = dad_edges.get(edge.innovation)

            if b is None or m is None or d is None or not edge.enabled:
                genotype.edges.append(Edge(edge.config, edge.input, edge.output, edge.enabled, edge.innovation, mutable=edge.enabled, weight=edge.weight))
            else:
                genotype.edges.append(Edge(edge.config, edge.input, edge.output, edge.enabled, edge.innovation, mutable=False, weight=c + (b - c) * f + (m - d) * f))

        return genotype

    @staticmethod
    def crossover_single(ancestor: "Genotype", config: Config, innovation_map: InnovationMap) -> "Genotype":
        genotype = Genotype(ancestor)
        genotype.nodes = [Node(node.id, node.type) for node in ancestor.nodes]
        genotype.edges = [Edge(edge.config, edge.input, edge.output, edge.enabled, edge.innovation, weight=edge.weight, mutable=edge.mutable) for edge in ancestor.edges]

        if np.random.uniform() < config.compatibility_low_node_over_edge:
            genotype.mutate_add_node(config, innovation_map)
        else:
            genotype.mutate_add_edge(config, innovation_map)

        return genotype

    def is_compatible(self, other: "Genotype", threshold: float, max_diffs: int) -> Tuple[bool, float]:
        return self.__is_compatible(self, other, threshold, max_diffs)

    @staticmethod
    def __is_compatible(g0: "Genotype", g1: "Genotype", threshold: float, max_diffs: int) -> Tuple[bool, float]:
        if g0.edges[-1].innovation < g1.edges[-1].innovation:
            g0, g1 = g1, g0

        m = 0  # Matching genes
        d = 0  # Disjoint genes
        w = 0.0  # Weight difference of matching genes

        g0_counter, g1_counter = 0, 0

        while g0_counter < len(g0.edges) and g1_counter < len(g1.edges):
            g0_edge = g0.edges[g0_counter]
            g1_edge = g1.edges[g1_counter]

            if g0_edge.innovation == g1_edge.innovation:
                g0_counter += 1
                g1_counter += 1
                m += 1
                w += min(1, abs(g0_edge.weight - g1_edge.weight))
            elif g0_edge.innovation < g1_edge.innovation:
                g0_counter += 1
                d += 1
            elif g0_edge.innovation > g1_edge.innovation:
                g1_counter += 1
                d += 1

        e = len(g0.edges) - g0_counter  # Excess genes

        topology_compatibility = m / max(len(g0.edges), len(g1.edges))
        return topology_compatibility >= threshold or e + d <= max_diffs, topology_compatibility

    def __str__(self):
        return "Score: " + str(self.score) + "; Nodes: " + str(sum(1 for n in self.nodes if n.is_hidden() or n.is_output())) + "; Edges: " + str(sum([1 for edge in self.edges if edge.enabled]))

    def __repr__(self):
        return "Score: " + str(self.score) + "; Nodes: " + str(sum(1 for n in self.nodes if n.is_hidden() or n.is_output())) + "; Edges: " + str(sum([1 for edge in self.edges if edge.enabled]))
