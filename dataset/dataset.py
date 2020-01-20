from typing import Any, Tuple, List, Union

from neat.ann.ann import Ann


class Dataset:

    """Abstract class for datasets implementation
    """
    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_bias_size(self) -> int:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_fitness(self, ann: Ann, seed: Any = None) -> Tuple[float, Union[None, List[Any]], List[float]]:
        raise NotImplementedError()

    def render(self, ann: Ann, seed: Any = None, **kwargs) -> None:
        raise NotImplementedError()
