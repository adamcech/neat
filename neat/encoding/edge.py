import numpy as np


class Edge:
    """Genome representation of connection genes
    """
    def __init__(self, input_node: int, output_node: int, enabled: bool, innovation: int, **kwargs):
        self.input = input_node
        self.output = output_node
        self.enabled = enabled
        self.innovation = innovation

        mutable = kwargs.get("mutable", None)
        self.mutable = True if mutable is None else mutable

        self.weight = kwargs.get("weight", None)
        self.mutate_random_weight() if self.weight is None else self.set_weight(self.weight)

    def mutate_random_weight(self):
        # self.weight = np.random.normal(0.0, 0.25)
        self.weight = np.random.normal(0.0, 1.0)

    def mutate_perturbate_weight(self):
        # self.set_weight(self.weight + np.random.normal(0, 0.1))
        self.set_weight(self.weight + np.random.normal(0, 0.2))

    def mutate_shift_weight(self):
        # self.set_weight(self.weight + np.random.normal(0.0, 0.05))
        self.set_weight(self.weight * np.random.uniform(0.95, 1.05))

    def set_weight(self, weight: float):
        if weight > 8 or weight < -8:
            self.mutate_random_weight()
        else:
            self.weight = weight

    def __repr__(self):
        """
        return "Edge(" + str(self.input.id) + ", " + str(self.output.id) + ", " + str(self.weight) + ", " + \
               str(self.enabled) + ", " + str(self.innovation) + ")"
        """
        return str(self.input) + "->" + str(self.output) + " (" + str(self.innovation) + ") " + str(self.weight) if self.enabled else ""

    def __str__(self):
        return str(self.input) + "->" + str(self.output) + ": Weight: " + str(self.weight) + " Innovation: " + \
               str(self.innovation) + " Enabled: " + str(self.enabled)
