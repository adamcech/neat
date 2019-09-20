import numpy as np
from neat.encoding.node import Node


class Edge:
    """Genome representation of connection genes
    """

    __innovation_counter = -1

    @staticmethod
    def next_innovation_value():
        Edge.__innovation_counter += 1
        return Edge.__innovation_counter

    def __init__(self, input_node: Node, output_node: Node):
        self.input = input_node.id
        self.output = output_node.id

        self.weight = np.random.uniform(-1, 1)
        self.enabled = True
        self.innovation = Edge.next_innovation_value()

    def mutate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def __repr__(self):
        return "Connection(" + str(self.input) + ", " + str(self.output) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ")"

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
