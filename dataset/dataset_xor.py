from collections import Generator

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
        self._generator = self._generator()

    def get_input_size(self) -> int:
        return 2

    def get_output_size(self) -> int:
        return 1

    def get_dataset_size(self) -> int:
        return 4

    def next_item(self) -> DatasetItem:
        return next(self._generator)

    def _generator(self):
        while True:
            for i in range(4):
                yield self._dataset[i]