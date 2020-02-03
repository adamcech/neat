import numpy as np

from neat.config import Config


class Edge:
    """Genome representation of connection genes
    """
    def __init__(self, config: Config, input_node: int, output_node: int, enabled: bool, innovation: int, **kwargs):
        self.config = config

        self.input = input_node
        self.output = output_node
        self.enabled = enabled
        self.innovation = innovation

        mutable = kwargs.get("mutable", None)
        self.mutable = True if mutable is None else mutable

        self.weight = kwargs.get("weight", None)
        self.mutate_random_weight() if self.weight is None else self.set_weight(self.weight)

    def mutate_random_weight(self):
        self.set_weight(np.random.normal(self.config.mutate_random_weight_mu, self.config.mutate_random_weight_sigm))

    def mutate_perturbate_weight(self):
        self.set_weight(self.weight + np.random.normal(self.config.mutate_perturbate_weight_mu, self.config.mutate_perturbate_weight_sigm))

    def mutate_shift_weight(self):
        self.set_weight(self.weight * np.random.uniform(self.config.mutate_shift_weight_lower_bound, self.config.mutate_shift_weight_upper_bound))

    def set_weight(self, weight: float):
        if weight < self.config.min_weight or weight > self.config.max_weight:
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
