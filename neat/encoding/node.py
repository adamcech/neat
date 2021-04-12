import random

from neat.encoding.node_type import NodeType
from neat.encoding.node_activation import NodeActivation


activations = [NodeActivation.TANH, NodeActivation.SIGM, NodeActivation.RELU, NodeActivation.ABS, NodeActivation.CLAMPED]


class Node:
    """Genome representation of node genes
    """

    def __init__(self, node_id: int, node_type: NodeType, activation: NodeActivation = None, output_id: int = None):
        self.id = node_id
        self.type = node_type

        self.output_id = output_id

        if activation is None:
            self.activation = random.sample(activations, 1)[0]
        else:
            self.activation = activation

    def mutate_activation(self):
        if len(activations) >= 2:
            self.activation = random.sample([x for x in activations if x != self.activation], 1)[0]

    def copy(self):
        return Node(self.id, self.type, self.activation, self.output_id)

    def is_input(self):
        return self.type == NodeType.INPUT

    def is_output(self):
        return self.type == NodeType.OUTPUT

    def is_hidden(self):
        return self.type == NodeType.HIDDEN

    def is_bias(self):
        return self.type == NodeType.BIAS

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "id = " + str(self.id) + "; layer = " + "; type =  " + str(self.type) + ";" + " act = " + str(self.activation)
