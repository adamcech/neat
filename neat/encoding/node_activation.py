from enum import Enum


class NodeActivation(Enum):
    """Representation for different activation functions of nodes"""
    CLAMPED = 1
    TANH = 2
    STEEPENED_TANH = 3
    SIGM = 4
    STEEPENED_SIGM = 5
