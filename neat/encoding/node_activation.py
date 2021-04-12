from enum import Enum


class NodeActivation(Enum):

    """Representation for different activation functions of nodes"""
    CLAMPED = 1
    TANH = 2
    STEEPENED_TANH = 3
    SIGM = 4
    STEEPENED_SIGM = 5
    RELU = 6
    INV = 7
    SIN = 8
    COS = 9
    ABS = 10
    STEP = 11
    LIN = 12
    GAUSS = 13
    SIGNUM = 14
