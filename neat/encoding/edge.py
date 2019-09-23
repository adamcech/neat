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

    def __init__(self, input_node: int, output_node: int, **kwargs):
        self.input = input_node
        self.output = output_node

        enabled = kwargs.get("enabled", None)
        self.enabled = True if enabled is None else enabled

        innovation = kwargs.get("innovation", None)
        self.innovation = -2 if innovation is None else innovation

        weight = kwargs.get("weight", None)
        self.weight = np.random.uniform(-1, 1) if weight is None else weight

    def mutate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def mutate_perturbate_weight(self):
        self.weight += np.random.normal(0.0, 0.5)

    def __repr__(self):
        """
        return "Edge(" + str(self.input.id) + ", " + str(self.output.id) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ")"
        """
        return str(self.input) + "->" + str(self.output) + " (" + str(self.innovation) + ")" if self.enabled else ""

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
