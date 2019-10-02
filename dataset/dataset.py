class Dataset:
    """Abstract class for datasets implementation
    """
    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_fitness(self, ann) -> float:
        raise NotImplementedError()

    def render(self, ann, **kwargs) -> None:
        raise NotImplementedError()
