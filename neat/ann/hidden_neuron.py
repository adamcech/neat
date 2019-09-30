from typing import List

import numpy as np

from neat.ann.connection import Connection
from neat.ann.neuron import Neuron


class HiddenNeuron(Neuron):
    """Node of ANN
    """
    def __init__(self, neuron_id: int, ann: "Ann"):
        super().__init__(neuron_id)

        self._done = False
        self._ann = ann
        self.connections = []  # type: List[Connection]

    def get_output(self):
        if self._done != self._ann.current_final_state:
            self._done = not self._done
            self._output = 1/(1 + np.power(np.e, -4.9 * sum([connection.forward() for connection in self.connections])))
            # self._output = 1 / (1 + np.power(np.e, -sum([connection.forward() for connection in self.connections])))

        return self._output
