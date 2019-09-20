from typing import List


class DatasetItem:
    """Class for representation of dataset item.
    """

    def __init__(self, item_input: List[float], item_output: List[float]):
        self.input = item_input
        self.output = item_output

    def __str__(self):
        return "Input: " + str(self.input) + "; Output: " + str(self.output)
