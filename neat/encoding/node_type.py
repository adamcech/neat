from enum import Enum


class NodeType(Enum):
    """Representation for different types of nodes"""
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3
