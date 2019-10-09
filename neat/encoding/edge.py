import numpy as np


class Edge:
    """Genome representation of connection genes
    """
    def __init__(self, input_node: int, output_node: int, enabled: bool, innovation: int, **kwargs):
        self.input = input_node
        self.output = output_node
        self.enabled = enabled
        self.innovation = innovation

        weight = kwargs.get("weight", None)
        self.weight = np.random.uniform(-1, 1) if weight is None else weight

    def mutate_random_weight(self):
        self.weight = np.random.uniform(-1, 1)

    def mutate_perturbate_weight(self):
        self.weight += np.random.normal(0, 1)

        if self.weight > 8:
            self.weight = 8
        elif self.weight < -8:
            self.weight = -8

    def __repr__(self):
        """
        return "Edge(" + str(self.input.id) + ", " + str(self.output.id) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ")"
        """
        return str(self.input) + "->" + str(self.output) + " (" + str(self.innovation) + ") " + str(self.weight) if self.enabled else ""

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
