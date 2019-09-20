from dataset.dataset import Dataset
from neat.encoding.connection import Connection
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """
    def __init__(self, dataset: Dataset):
        self._nodes = []
        self._connections = []

        self._fitness = 0

        # Adding input nodes
        for i in range(dataset.get_input_size()):
            self._nodes.append(Node(self._next_node_id(), NodeType.INPUT))

        # Adding output nodes
        for i in range(dataset.get_output_size()):
            self._nodes.append(Node(self._next_node_id(), NodeType.OUTPUT))

        # Connecting inputs to outputs
        for input_node in self._nodes:
            if input_node.is_input():
                for output_node in self._nodes:
                    if output_node.is_output():
                        self.add_connection(input_node, output_node)

    def create_node(self) -> Node:
        node = Node(self._next_node_id(), NodeType.HIDDEN)
        self._nodes.append(node)
        return node

    def add_connection(self, input_node: Node, output_node: Node):
        self._connections.append(Connection(input_node, output_node))

    def mutate_change_all_weights(self):
        for connection in self._connections:
            connection.mutate_random_weight()

    def _next_node_id(self) -> int:
        return len(self._nodes)

    def calculate(self) -> float:
        pass

    def get_fitness(self) -> float:
        return self._fitness

    def reset_fitness(self):
        self._fitness = 0

    def __str__(self):
        return str(self._nodes) + "\n" + str(self._connections)
