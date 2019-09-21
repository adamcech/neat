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

    def __init__(self, input_node: Node, output_node: Node, **kwargs):
        self.input = input_node
        self.output = output_node

        weight = kwargs.get("weight", None)
        self.weight = np.random.uniform(-1, 1) if weight is None else weight

        self.enabled = True
        self.innovation = Edge.next_innovation_value()

    def mutate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def mutate_perturbate_weight(self):
        self.weight += np.random.normal(0.0, 0.5)

    def __repr__(self):
        """
        return "Edge(" + str(self.input.id) + ", " + str(self.output.id) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ")"
        """
        return str(self.input.id) + "->" + str(self.output.id) if self.enabled else ""

    def __str__(self):
        return str(self.input.id) + "->" + str(self.output.id) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
