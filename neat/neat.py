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
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3
        self._t = t
        self._population_size = population_size
        self._dataset = dataset

        self._population = Population(population_size, self._dataset)

    def next_generation(self):
        pass

    def get_best_genotype(self) -> Genotype:
        return self._population.get_best()
