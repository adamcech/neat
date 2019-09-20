from typing import List

from dataset.dataset import Dataset
from neat.encoding.edge import Edge
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """
    def __init__(self, dataset: Dataset):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

        self._fitness = 0

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
                        self.add_edge(input_node, output_node)

    def add_node(self) -> Node:
        node = Node(self._next_node_id(), NodeType.HIDDEN)
        self.nodes.append(node)
        return node

    def add_edge(self, input_node: Node, output_node: Node):
        self.edges.append(Edge(input_node, output_node))

    def mutate_change_all_weights(self):
        for connection in self.edges:
            connection.mutate_random_weight()

    def reset_fitness(self):
        self._fitness = 0

    def get_fitness(self) -> float:
        return self._fitness

    def _next_node_id(self) -> int:
        return len(self.nodes)

    def __str__(self):
        return str(self.nodes) + "\n" + str(self.edges)
