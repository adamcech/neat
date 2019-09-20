from neat.ann.neuron import Neuron


class Connection:
    """Connection of ANN
    """
    def __init__(self, weight: float, from_neuron: Neuron):
        self.weight = weight
        self.from_neuron = from_neuron

    def forward(self):
        return self.from_neuron.get_output() * self.weight
