from dataset.dataset import Dataset
from neat.encoding.genotype import Genotype
import copy


class Population:
    """Population representation
    """

    def __init__(self, size, dataset: Dataset):
        self._size = size
        self._population = []

        initial_genotype = Genotype(dataset)

        for i in range(self._size):
            copied_genotype = copy.deepcopy(initial_genotype)
            copied_genotype.mutate_change_all_weights()
            self._population.append(copied_genotype)

    def get_best(self) -> Genotype:
        best_fitness = 0
        for i in range(1, len(self._population)):
            if self._population[best_fitness].get_fitness() < self._population[i].get_fitness():
                best_fitness = i

        return self._population[best_fitness]
