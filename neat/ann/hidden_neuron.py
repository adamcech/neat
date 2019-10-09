from typing import List

import numpy as np

from math import tanh

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

            # Tanh
            # self._output = tanh(sum([connection.forward() for connection in self.connections]))

            summation = sum([connection.forward() for connection in self.connections])

            # Steepened sigmoid for XOR
            self._output = 0 if summation < -144 else 1/(1 + np.power(np.e, 4.9 * -summation))

            # Sigmoid
            # self._output = 0 if summation < -709 else 1/(1 + np.power(np.e, -summation))

        return self._output
