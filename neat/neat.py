from dataset.dataset import Dataset
from neat.encoding.genotype import Genotype
from neat.population import Population


class Neat:
    """Neuroevolution of augmenting topologies

    Args:
         c1 (float): Excess Genes Importance
         c2 (float): Disjoint Genes Importance
         c3 (float): Weights difference Importance
         t (float): Compatibility Threshold
         population_size (int): Population Size
         dataset (Dataset): Dataset to use
    """

    def __init__(self, c1: float, c2: float, c3: float, t: float, population_size: int, dataset: Dataset):
        self._dataset = dataset
        self._population = Population(c1, c2, c3, t, population_size, self._dataset)

    def next_generations(self, generations: int):
        for i in range(1, generations + 1):
            self._next_generation()
            # self._population.print_all_fitness()
            print("Generation: " + str(i) + "; Best fitness: " + str(self.get_best_genotype().get_fitness()) + ";")

    def _next_generation(self):
        self._population.crossover()
        self._population.mutate_weights()
        self._population.mutate_add_edge()
        self._population.mutate_add_node()
        self._population.evaluate(self._dataset)

    def get_best_genotype(self) -> Genotype:
        return self._population.get_best()
