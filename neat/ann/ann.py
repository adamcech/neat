from typing import List, Union

import numpy

from neat.ann.connection import Connection
from neat.ann.neuron import Neuron
from neat.ann.hidden_neuron import HiddenNeuron
from neat.encoding.node_activation import NodeActivation


class Ann:
    """Genotype to ANN mapping
    """
    def __init__(self, genotype: "Genotype", node_activation: NodeActivation):
        self._inputs = []  # type: List[Neuron]
        self._hidden = []  # type: List[HiddenNeuron]
        self._outputs = []  # type: List[HiddenNeuron]

        self.current_final_state = False

        for node in genotype.nodes:
            if node.is_input() or node.is_bias():
                self._inputs.append(Neuron(node.id))
            elif node.is_hidden():
                self._hidden.append(HiddenNeuron(node.id, self, node_activation))
            elif node.is_output():
                self._outputs.append(HiddenNeuron(node.id, self, node_activation))

        for edge in [edge for edge in genotype.edges if edge.enabled]:
            for l in [self._hidden, self._outputs]:
                for output_neuron in l:
                    if edge.output == output_neuron.id:
                        for ll in [self._inputs, self._hidden]:
                            for input_neuron in ll:
                                if edge.input == input_neuron.id:
                                    output_neuron.connections.append(Connection(edge.weight, input_neuron))
                                    break

    def calculate(self, item_input: Union[numpy.ndarray, List[float]]) -> List[float]:
        self.current_final_state = not self.current_final_state

        for i in range(len(self._inputs)):
            self._inputs[i].set_output(item_input[i])

        return [output.get_output() for output in self._outputs]
