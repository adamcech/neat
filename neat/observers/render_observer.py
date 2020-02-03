from neat.ann.ann import Ann
from neat.neat import Neat

from neat.observers.abstract_observer import AbstractObserver


class RenderObserver(AbstractObserver):

    def __init__(self, render_counter=25):
        self._generation = 0
        self._render_counter = render_counter

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def end_generation(self, neat: Neat) -> None:
        if self._generation % self._render_counter == self._render_counter - 1 and self._generation != 0:
            genotype = neat.population.get_best_member()
            print(genotype)
            neat.config.dataset.render(Ann(genotype, neat.config.activation), genotype.evaluated_seed, loops=1)
