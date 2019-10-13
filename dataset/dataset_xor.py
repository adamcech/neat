from collections import Generator

import numpy as np

from dataset.dataset import Dataset
from dataset.dataset_item import DatasetItem


class DatasetXor(Dataset):
    """XOR Generator
    0 0: 0
    0 1: 1
    1 0: 1
    1 1: 0
    """

    def __init__(self):
        xor00 = DatasetItem([0, 0], [0])
        xor01 = DatasetItem([0, 1], [1])
        xor10 = DatasetItem([1, 0], [1])
        xor11 = DatasetItem([1, 1], [0])
        self._dataset = [xor00, xor01, xor10, xor11]

        bias_input = [1 for _ in range(self.get_bias_size())]
        for item in self._dataset:
            item.input += bias_input

    def get_input_size(self) -> int:
        return 2

    def get_bias_size(self) -> int:
        return 1

    def get_output_size(self) -> int:
        return 1

    def get_fitness(self, ann) -> float:
        fitness = 0

        for item in self._dataset:
            result = ann.calculate(item.input)
            fitness += np.abs(sum([item.output[i] - result[i] for i in range(self.get_output_size())]))

        return np.power(4 - fitness, 2)

    def get_dataset(self):
        return self._dataset

    def render(self, ann: "Ann", **kwargs) -> None:
        for item in self._dataset:
            result = ann.calculate(item.input)
            print(str(item) + "; Result " + str(result))

