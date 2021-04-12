from enum import Enum


class TargetFunction(Enum):
    CLAMPED = 1
    SOFTMAX = 2
    STEEPENED_TANH = 3
    TANH = 4
    STEEPENED_SIGM = 5
    SIGM = 6
    RELU = 7
    LIN = 8

