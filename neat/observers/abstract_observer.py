from neat.population import Population


class AbstractObserver:

    def start_generation(self, generation: int) -> None:
        raise NotImplementedError()

    def end_generation(self, population: Population) -> None:
        raise NotImplementedError()
