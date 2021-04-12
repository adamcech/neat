from typing import List
from math import e

from neat.ann.connection import Connection
from neat.ann.neuron import Neuron
from neat.encoding.node_activation import NodeActivation


class HiddenNeuron(Neuron):
    """Node of ANN
    """
    def __init__(self, neuron_id: int, ann, activation: NodeActivation, connections: List[Connection] = None):
        super().__init__(neuron_id)

        if connections is None:
            connections = []

        self._done = False
        self._ann = ann
        self.connections = connections

        self.activation_function = activation

        if self.activation_function == NodeActivation.CLAMPED:
            self.activation = self.__clamped
        elif self.activation_function == NodeActivation.SIGM:
            self.activation = self.__sigm
        elif self.activation_function == NodeActivation.STEEPENED_SIGM:
            self.activation = self.__steepened_sigm
        elif self.activation_function == NodeActivation.TANH:
            self.activation = self.__tanh
        elif self.activation_function == NodeActivation.STEEPENED_TANH:
            self.activation = self.__steepened_tanh
        else:
            raise Exception("Unknown activation function")

    def __reduce__(self):
        return self.__class__, (self.id, self._ann, self.activation_function, self.connections)

    def get_output(self):
        if self._done != self._ann.current_final_state:
            self._done = not self._done
            self._output = self.activation(sum([connection.forward() for connection in self.connections]))
        return self._output

    def __clamped(self, x: float) -> float:
        return -1 if x < -1 else (1 if x > 1 else x)

    def __tanh(self, x: float) -> float:
        return -1 if x < -354 else 2 / (1 + e ** (-2*x)) - 1

    def __steepened_tanh(self, x: float) -> float:
        return -1 if x < -72 else 2 / (1 + e ** -(9.8*x)) - 1

    def __sigm(self, x: float) -> float:
        return 0 if x < -709 else 1/ (1 + e ** -x)

    def __steepened_sigm(self, x: float) -> float:
        return 0 if x < -144 else 1 / (1 + e ** (-4.9 * x))
