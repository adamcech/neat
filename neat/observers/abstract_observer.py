class AbstractObserver:

    def start_generation(self, generation: int) -> None:
        raise NotImplementedError()

    def end_generation(self, neat: "Neat") -> None:
        raise NotImplementedError()
