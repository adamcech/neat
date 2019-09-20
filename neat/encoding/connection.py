import numpy as np
from neat.encoding.node import Node


class Connection:
    """Genome representation of connection genes
    """

    __innovation_counter = -1

    @staticmethod
    def next_innovation_value():
        Connection.__innovation_counter += 1
        return Connection.__innovation_counter

    def __init__(self, input_node: Node, output_node: Node, weight=np.random.uniform(-1, 1)):
        self.input = input_node.id
        self.output = output_node.id
        self.weight = weight

        self.enabled = True
        self.innovation = Connection.next_innovation_value()

    def mutate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def __repr__(self):
        return "Connection(" + str(self.input) + ", " + str(self.output) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ", " + str(id(self)) + ")"

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
