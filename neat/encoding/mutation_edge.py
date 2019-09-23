from neat.encoding.edge import Edge


class MutationEdge:
    """Keeping track of edges mutations

    Args:
        new_edge (Edge): Added edge
    """

    def __init__(self, new_edge: Edge):
        self.new_edge = new_edge
