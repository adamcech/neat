from dataset.dataset import Dataset
from neat.ann.ann import Ann
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
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.t = t
        self.population_size = population_size
        self.dataset = dataset
        self._population = Population(self)

        self.best_genotype = None

    def next_generations(self, generations: int, **kwargs):
        output = kwargs.get("output", None)
        render = kwargs.get("render", None)

        for i in range(1, generations + 1):
            self._next_generation()

            curr_best = self._population.get_best()

            if self.best_genotype is None:
                self.best_genotype = curr_best

            if curr_best.fitness > self.best_genotype.fitness:
                self.best_genotype = curr_best.deepcopy()

            if output:
                print(str(i) + ":  \t" + str(self._population.get_best().fitness) + " \t\t" + str(self.get_best_genotype().fitness) + "; \tSpecies: " + str(len(self._population.get_species())) + " " + str(self._population.get_species()))

            if render is not None:
                if render:
                    print(self.get_best_genotype())
                    self.dataset.render(Ann(self.get_best_genotype()), loops=3)

    def _next_generation(self):
        self._population.speciate()
        self._population.crossover()
        self._population.mutate_weights()
        self._population.mutate_add_edge()
        self._population.mutate_add_node()
        self._population.evaluate(self.dataset)

    def get_best_genotype(self) -> Genotype:
        return self.best_genotype
