import copy
import random
from typing import List

import numpy as np

from dataset.dataset import Dataset
from neat.encoding.innovation_map import InnovationMap
from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """

    __innovation_map = InnovationMap()

    def __init__(self):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]
        self.fitness = 0  # float
        self.score = 0  # float

    @staticmethod
    def initial_genotype(dataset: Dataset) -> "Genotype":
        genotype = Genotype()

        # Adding input nodes
        for i in range(dataset.get_input_size()):
            genotype.nodes.append(Node(Genotype.__innovation_map.next_innovation(), NodeType.INPUT))

        # Adding bias nodes
        for i in range(dataset.get_bias_size()):
            genotype.nodes.append(Node(Genotype.__innovation_map.next_innovation(), NodeType.BIAS))

        # Adding output nodes
        for i in range(dataset.get_output_size()):
            genotype.nodes.append(Node(Genotype.__innovation_map.next_innovation(), NodeType.OUTPUT))

        """
        # Connecting bias to outputs
        for bias_node in [node for node in genotype.nodes if node.is_bias()]:
            for output_node in [node for node in genotype.nodes if node.is_output()]:
                genotype.edges.append(
                    Edge(bias_node.id, output_node.id, True,
                         Genotype.__innovation_map.get_edge_innovation(bias_node.id, output_node.id)))
        """

        # Connecting inputs and bias to outputs
        for input_node in [node for node in genotype.nodes if node.is_input() or node.is_bias()]:
            for output_node in [node for node in genotype.nodes if node.is_output()]:
                genotype.edges.append(
                    Edge(input_node.id, output_node.id, True,
                         Genotype.__innovation_map.get_edge_innovation(input_node.id, output_node.id)))

        return genotype

    def initial_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for connection in self.edges:
            connection.mutate_random_weight()
        return cp

    def deepcopy(self) -> "Genotype":
        return copy.deepcopy(self)

    def mutate_add_node(self):
        old_edge = random.sample(self.edges, 1)[0]  # type: Edge
        old_edge.enabled = False

        new_node = Node(Genotype.__innovation_map.get_node_innovation(old_edge.input, old_edge.output, self.nodes), NodeType.HIDDEN)
        new_edge = Edge(old_edge.input, new_node.id, True, Genotype.__innovation_map.get_edge_innovation(old_edge.input, new_node.id), weight=1)
        original_edge = Edge(new_node.id, old_edge.output, True, Genotype.__innovation_map.get_edge_innovation(new_node.id, old_edge.output), weight=old_edge.weight)

        self.nodes.append(new_node)
        self.edges.append(new_edge)
        self.edges.append(original_edge)

    def mutate_add_edge(self):
        for try_counter in range(0, 100):
            samples = random.sample(self.nodes, 2)
            input_node = samples[0]  # type: Node
            output_node = samples[1]  # type: Node
            if (input_node.is_output() or (output_node.is_input() or output_node.is_bias())) or \
                    ((input_node.is_input() or input_node.is_bias()) and (output_node.is_input() or input_node.is_bias())) or \
                    (input_node.is_output() and output_node.is_output()) or \
                    any((edge.input == input_node.id and edge.output == output_node.id) or
                        (edge.input == output_node.id and edge.output == input_node.id) for edge in self.edges):
                continue

            if not self._is_new_edge_recurrent(input_node.id, output_node.id):
                self.edges.append(
                    Edge(input_node.id,
                         output_node.id,
                         True,
                         Genotype.__innovation_map.get_edge_innovation(input_node.id, output_node.id)))
                break

    def _is_new_edge_recurrent(self, input_node: int, output_node: int) -> bool:
        # Search for path from output_node to input_node
        # TODO better solution
        return input_node in self._dfs(output_node)

    def _dfs(self, input_node: int) -> List[int]:
        visited = []
        self._dfs_step(input_node, visited)
        return visited

    def _dfs_step(self, start_node: int, visited: List[int]):
        visited.append(start_node)

        for node in self._get_adjacent_nodes(start_node):
            if node not in visited:
                self._dfs_step(node, visited)

    def _get_adjacent_nodes(self, node_id: int) -> List[int]:
        nodes = []
        for edge in self.edges:
            if edge.input == node_id and edge.output not in nodes:
                nodes.append(edge.output)
        return nodes

    def calculate_fitness(self, dataset: Dataset):
        self.fitness = dataset.get_fitness(Ann(self))

    def __str__(self):
        return str(self.fitness) + "\n" + \
               str(len(self.nodes)) + " " + str(self.nodes) + "\n" + \
               str(len(self.edges)) + " " + str(self.edges)

    @staticmethod
    def crossover(mom: "Genotype", dad: "Genotype") -> "Genotype":
        better = mom if mom.fitness > dad.fitness else dad
        worse = mom if mom.fitness <= dad.fitness else dad

        genotype = Genotype()

        for node in better.nodes:
            genotype.nodes.append(Node(node.id, node.type))

        better_counter, worse_counter = 0, 0

        while better_counter < len(better.edges) and worse_counter < len(worse.edges):
            skip = False
            b = better.edges[better_counter]
            w = worse.edges[worse_counter]
            selected = b

            if b.innovation == w.innovation:
                better_counter += 1
                worse_counter += 1
                if np.random.uniform(0, 1) < 0.5:
                    selected = w
            elif b.innovation < w.innovation:
                better_counter += 1
            elif b.innovation > w.innovation:
                worse_counter += 1
                skip = True

            if not skip:
                genotype._add_edge_from_crossover(selected)

        for i in range(better_counter, len(better.edges)):
            genotype._add_edge_from_crossover(better.edges[i])

        return genotype

    def _add_edge_from_crossover(self, edge: Edge):
        self.edges.append(Edge(edge.input, edge.output, edge.enabled, edge.innovation, weight=edge.weight))

    @staticmethod
    def is_compatible(neat: "Neat", g0: "Genotype", g1: "Genotype"):
        if g0.edges[-1].innovation < g1.edges[-1].innovation:
            g0, g1 = g1, g0

        g0_n = len(g0.edges)
        g1_n = len(g1.edges)
        n = 1 if g0_n < 20 and g1_n < 20 else max(g0_n, g1_n)  # Number of genes in larger genotype, 1 if small genotype

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
                w += abs(g0_edge.weight - g1_edge.weight)
            elif g0_edge.innovation < g1_edge.innovation:
                g0_counter += 1
                d += 1
            elif g0_edge.innovation > g1_edge.innovation:
                g1_counter += 1
                d += 1

        e = len(g0.edges) - g0_counter  # Excess genes

        return neat.t > neat.c1 * e / n + neat.c2 * d / n + neat.c3 * w / m

    def __repr__(self):
        return str(self.fitness)
