import copy
import random
from typing import List

import numpy as np

from dataset.dataset import Dataset
from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.mutation_edge import MutationEdge
from neat.encoding.mutation_node import MutationNode
from neat.encoding.node import Node
from neat.encoding.node_id_map import NodeIdMap
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """

    __node_id_map = NodeIdMap()

    @staticmethod
    def shit():
        return Genotype.__node_id_map

    def __init__(self):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

        self._fitness = 0
        self._shared_fitness = 0

    @staticmethod
    def initial_genotype(dataset: Dataset) -> "Genotype":
        genotype = Genotype()

        # Adding input nodes
        for i in range(dataset.get_input_size()):
            genotype.nodes.append(Node(Genotype.__node_id_map.next_id(), NodeType.INPUT))

        # Adding output nodes
        for i in range(dataset.get_output_size()):
            genotype.nodes.append(Node(Genotype.__node_id_map.next_id(), NodeType.OUTPUT))

        # Connecting inputs to outputs
        for input_node in genotype.nodes:
            if input_node.is_input():
                for output_node in genotype.nodes:
                    if output_node.is_output():
                        genotype._add_edge(genotype._create_edge(input_node.id, output_node.id))

        return genotype

    def mutate_add_node(self, mutations: List[MutationNode]):
        old_edge = random.sample(self.edges, 1)[0]
        old_edge.enabled = False

        new_node = self._create_node(old_edge.input, old_edge.output)

        original_edge = self._create_edge(old_edge.input, new_node.id, weight=old_edge.weight)
        new_edge = self._create_edge(new_node.id, old_edge.output, weight=1)

        new_innovation = True
        same_mutation = None
        for mutation in mutations:
            if mutation.old_edge.innovation == old_edge.innovation:
                new_innovation = False
                same_mutation = mutation
                break

        self._add_node(new_node)

        if new_innovation:
            self._add_edge(original_edge)
            self._add_edge(new_edge)
            mutations.append(MutationNode(old_edge, new_node, original_edge, new_edge))
        else:
            self._add_edge(original_edge, innovation=same_mutation.original_edge.innovation)
            self._add_edge(new_edge, innovation=same_mutation.new_edge.innovation)

    def mutate_add_edge(self, mutations: List[MutationEdge]):
        for try_counter in range(0, 10):
            samples = random.sample(self.nodes, 2)
            input_node = samples[0]  # type: Node
            output_node = samples[1]  # type: Node
            if input_node.type == NodeType.OUTPUT or output_node.type == NodeType.INPUT or \
                    (input_node.type == NodeType.INPUT and output_node.type == NodeType.INPUT) or \
                    (input_node.type == NodeType.OUTPUT and output_node.type == NodeType.OUTPUT) or \
                    any((edge.input == input_node.id and edge.output == output_node.id) or
                        (edge.input == output_node.id and edge.output == input_node.id) for edge in self.edges):
                continue

            if not self._is_new_edge_recurrent(input_node.id, output_node.id):
                new_edge = self._create_edge(input_node.id, output_node.id)

                new_innovation = True
                same_mutation = None

                for mutation in mutations:
                    if mutation.new_edge.input == new_edge.input and mutation.new_edge.output == new_edge.output:
                        new_innovation = False
                        same_mutation = mutation
                        break

                if new_innovation:
                    self._add_edge(new_edge)
                    mutations.append(MutationEdge(new_edge))
                else:
                    self._add_edge(new_edge, innovation=same_mutation.new_edge.innovation)

                break

    def get_population_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for connection in self.edges:
            connection.mutate_random_weight()

        return cp

    def evaluate(self, dataset: Dataset):
        self._fitness = dataset.get_fitness(Ann(self))

    """
    def evaluate_shared_fitness(self, c1: float, c2: float, c3: float, species: List["Genotype"]):
        self._shared_fitness = self.get_fitness() / sum([Genotype.calculate_compatibility(c1, c2, c3, self, genotype)
                                                         for genotype in species if genotype != self])
    """

    def evaluate_shared_fitness(self, c1: float, c2: float, c3: float, t: float, species: List["Genotype"]):
        self._shared_fitness = self.get_fitness() / sum([
            1 if Genotype.calculate_compatibility(c1, c2, c3, self, other) < t else 0 for other in species])

    def get_shared_fitness(self):
        return self._shared_fitness

    def get_fitness(self) -> float:
        return self._fitness

    def _create_node(self, input_node: int, output_node: int) -> Node:
        return Node(Genotype.__node_id_map.get_id(input_node, output_node, self.nodes), NodeType.HIDDEN)

    def _create_edge(self, input_node: int, output_node: int, **kwargs) -> Edge:
        weight = kwargs.get("weight", None)
        return Edge(input_node, output_node, weight=weight)

    def _add_node(self, node: Node):
        self.nodes.append(node)

    def _add_edge(self, edge: Edge, **kwargs):
        innovation = kwargs.get("innovation", None)
        edge.innovation = Edge.next_innovation_value() if innovation is None else innovation
        self.edges.append(edge)

    def _is_new_edge_recurrent(self, input_node: int, output_node: int) -> bool:
        # Search for path from output_node to input_node
        # TODO better solution
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

    def __str__(self):
        return str(id(self)) + "\n" + str(self.nodes) + "\n" + str(self.edges)

    @staticmethod
    def crossover(better: "Genotype", worse: "Genotype") -> "Genotype":
        if worse.get_shared_fitness() > better.get_shared_fitness():
            tmp = worse
            worse = better
            better = tmp

        genotype = Genotype()

        for node in better.nodes:
            genotype.nodes.append(Node(node.id, node.type))

        for g0 in better.edges:
            selected = g0
            for g1 in worse.edges:
                if g0.innovation == g1.innovation:
                    if np.random.uniform(0, 1) < 0.5:
                        selected = g1
                    break

            genotype.edges.append(
                Edge(selected.input,
                     selected.output,
                     weight=selected.weight,
                     enabled=selected.enabled if selected.enabled else (np.random.uniform(0, 1) < 0.25),
                     innovation=selected.innovation))

        return genotype

    @staticmethod
    def calculate_compatibility(c1, c2, c3, g0: "Genotype", g1: "Genotype"):
        """Calculates compatibility between two genotypes

        Args:
            c1 (float): Excess Genes Importance
            c2 (float): Disjoint Genes Importance
            c3 (float): Weights difference Importance
            g0 (Genotype): Compatibility Threshold
            g1 (Genotype): Population size

        Returns:
            float: Compatibility for g0 and g1
        """
        if g0.edges[-1].innovation < g1.edges[-1].innovation:
            tmp = g0
            g1 = g0
            g0 = tmp

        g0_n = len(g0.edges)
        g1_n = len(g1.edges)

        n = 1 if g0_n < 20 and g1_n < 20 else max(g0_n, g1_n)
        # n = max(g0_n, g1_n)

        m = 0  # Matching genes
        e = 0  # Excess genes
        d = 0  # Disjoint genes
        w = 0.0  # Weight difference of matching genes

        g1_max_innovation = g1.edges[-1].innovation

        for g0_genome in g0.edges:
            gene_found = False
            for g1_genome in g1.edges:
                if g0_genome.innovation == g1_genome.innovation:
                    gene_found = True
                    w += np.abs(g0_genome.weight - g1_genome.weight)
                    m += 1
                    break

            if not gene_found:
                if g0_genome.innovation < g1_max_innovation:
                    d += 1
                else:
                    e += 1

        d += len(g1.edges) - m

        return c1 * e / n + c2 * d / n + c3 * w / m
