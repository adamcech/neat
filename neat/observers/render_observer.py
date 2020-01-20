from dataset.dataset import Dataset
from neat.ann.ann import Ann

from neat.observers.abstract_observer import AbstractObserver


class RenderObserver(AbstractObserver):

    def __init__(self, dataset: Dataset, render_counter=25):
        self._generation = 0
        self._dataset = dataset
        self._render_counter = render_counter

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def end_generation(self, neat: "Neat") -> None:
        population = neat.population

        if self._generation % self._render_counter == self._render_counter - 1 and self._generation != 0:
            self._render(population.get_best_member())

    def _render(self, genotype: "Genotype"):
        print(genotype)
        self._dataset.render(Ann(genotype), genotype.evaluated_seed, loops=1)
