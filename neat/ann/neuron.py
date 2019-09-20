class Neuron:
    """Node of ANN
    """
    def __init__(self, neuron_id: int):
        self.id = neuron_id
        self._output = 0

    def set_output(self, value: float):
        self._output = value

    def get_output(self):
        return self._output
