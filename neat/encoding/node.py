from neat.encoding.node_type import NodeType


class Node:
    """Genome representation of node genes
    """

    def __init__(self, node_id: int, node_type: NodeType):
        self.id = node_id
        self.type = node_type
        self.value = 0

    def is_input(self):
        return self.type == NodeType.INPUT

    def is_output(self):
        return self.type == NodeType.OUTPUT

    def is_hidden(self):
        return self.type == NodeType.HIDDEN

    def __repr__(self):
        return "Node(" + str(self.id) + ", " + str(self.type) + ", " + str(id(self)) + ")"

    def __str__(self):
        return str(self.id) + ": " + str(self.type)