from dataset.dataset_item import DatasetItem


class Dataset:
    """Abstract class for datasets implementation
    """
    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_dataset_size(self) -> int:
        raise NotImplementedError()

    def next_item(self) -> DatasetItem:
        raise NotImplementedError()
