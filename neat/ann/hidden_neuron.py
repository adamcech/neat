from typing import List
from math import e

from neat.ann.connection import Connection
from neat.ann.neuron import Neuron
from neat.encoding.node_activation import NodeActivation


class HiddenNeuron(Neuron):
    """Node of ANN
    """
    def __init__(self, neuron_id: int, ann: "Ann", activation: NodeActivation):
        super().__init__(neuron_id)

        self._done = False
        self._ann = ann
        self.connections = []  # type: List[Connection]

        if activation == NodeActivation.CLAMPED:
            self.activation = HiddenNeuron.__clamped
        elif activation == NodeActivation.SIGM:
            self.activation = HiddenNeuron.__sigm
        elif activation == NodeActivation.STEEPENED_SIGM:
            self.activation = HiddenNeuron.__steepened_sigm
        elif activation == NodeActivation.TANH:
            self.activation = HiddenNeuron.__tanh
        elif activation == NodeActivation.STEEPENED_TANH:
            self.activation = HiddenNeuron.__steepened_tanh
        else:
            raise Exception("Unknown activation function")

    # Steepened sigmoid for XOR
    def get_output(self):
        if self._done != self._ann.current_final_state:
            self._done = not self._done
            self._output = self.activation(sum([connection.forward() for connection in self.connections]))
        return self._output

    @staticmethod
    def __clamped(x: float) -> float:
        return -1 if x < -1 else (1 if x > 1 else x)

    @staticmethod
    def __tanh(x: float) -> float:
        return -1 if x < -354 else 2 / (1 + e ** (-2*x)) - 1

    @staticmethod
    def __steepened_tanh(x: float) -> float:
        return -1 if x < -72 else 2 / (1 + e ** -(9.8*x)) - 1

    @staticmethod
    def __sigm(x: float) -> float:
        return 0 if x < -709 else 1/ (1 + e ** -x)

    @staticmethod
    def __steepened_sigm(x: float) -> float:
        return 0 if x < -144 else 1 / (1 + e ** (-4.9 * x))
