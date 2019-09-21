import copy
import random
from typing import List

from dataset.dataset import Dataset
from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """
    def __init__(self, dataset: Dataset):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

        self._ann = None  # type: Ann
        self.fitness = 0

        # Adding input nodes
        for i in range(dataset.get_input_size()):
            self.nodes.append(Node(self._next_node_id(), NodeType.INPUT))

        # Adding output nodes
        for i in range(dataset.get_output_size()):
            self.nodes.append(Node(self._next_node_id(), NodeType.OUTPUT))

        # Connecting inputs to outputs
        for input_node in self.nodes:
            if input_node.is_input():
                for output_node in self.nodes:
                    if output_node.is_output():
                        self._add_edge(input_node, output_node)

    def mutate_add_node(self, old_edge: Edge):
        old_edge.enabled = False
        new_node = self._add_node()
        self._add_edge(old_edge.input, new_node, weight=1)
        self._add_edge(new_node, old_edge.output, weight=old_edge.weight)

    def mutate_add_edge(self):
        for try_counter in range(0, 10):
            samples = random.sample(self.nodes, 2)
            input_node = samples[0]  # type: Node
            output_node = samples[1]  # type: Node
            if input_node.type == NodeType.OUTPUT or output_node.type == NodeType.INPUT or \
               (input_node.type == NodeType.INPUT and output_node.type == NodeType.INPUT) or \
               (input_node.type == NodeType.OUTPUT and output_node.type == NodeType.OUTPUT) or \
               any((edge.input == input_node and edge.output == output_node) or (edge.input == output_node and edge.output == input_node) for edge in self.edges):
                continue

            if not self._is_edge_recurrent(input_node, output_node):
                self._add_edge(input_node, output_node)
                break

    def get_population_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for connection in self.edges:
            connection.mutate_random_weight()

        return cp

    def evaluate(self, dataset: Dataset):
        self._ann = Ann(self)
        self.fitness = dataset.get_fitness(self._ann)

    def get_fitness(self) -> float:
        return self.fitness

    def _add_node(self) -> Node:
        node = Node(self._next_node_id(), NodeType.HIDDEN)
        self.nodes.append(node)
        return node

    def _add_edge(self, input_node: Node, output_node: Node, **kwargs):
        weight = kwargs.get("weight", None)
        self.edges.append(Edge(input_node, output_node, weight=weight))

    def _is_edge_recurrent(self, input_node: Node, output_node: Node) -> bool:
        # Search for path from output_node to input_node
        return self._dfs(input_node, output_node, [])

    def _dfs(self, search_node: Node, start_node: Node, visited: List[Node]) -> bool:
        visited.append(start_node)

        for node in self._get_adjacent_nodes(start_node):
            if node == search_node:
                return True
            elif node not in visited:
                return self._dfs(search_node, node, visited)

        return False

    def _get_adjacent_nodes(self, node: Node) -> List[Node]:
        nodes = []
        for edge in self.edges:
            if edge.input == node and edge.output not in nodes:
                nodes.append(edge.output)
        return nodes

    def _next_node_id(self) -> int:
        return len(self.nodes)

    def __str__(self):
        return str(self.nodes) + "\n" + str(self.edges)
