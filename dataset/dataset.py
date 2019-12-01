from typing import Tuple


class Dataset:
    """Abstract class for datasets implementation
    """
    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_bias_size(self) -> int:
        raise NotImplementedError()

    def set_max_trials(self, max_trials: int) -> None:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_fitness(self, ann) -> float:
        raise NotImplementedError()

    def render(self, ann, **kwargs) -> None:
        raise NotImplementedError()
