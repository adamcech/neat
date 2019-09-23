from neat.encoding.edge import Edge
from neat.encoding.node import Node


class MutationNode:
    """Keeping track od nodes mutations

    Args:
        old_edge (Edge): Original edge to be replaced
        new_node (Node): New node to be placed on old edge
        original_edge (Edge): Original edge replacement (same weight)
        new_edge (Edge): New edge connecting new Node (weight 1)
    """
    def __init__(self, old_edge: Edge, new_node: Node, original_edge: Edge, new_edge: Edge):
        self.old_edge = old_edge
        self.new_node = new_node
        self.original_edge = original_edge
        self.new_edge = new_edge
