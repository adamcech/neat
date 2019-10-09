from dataset.dataset_xor import DatasetXor
from neat.ann.ann import Ann
from neat.neat import Neat

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3

population_size = 150
generations = 200

dataset = DatasetXor()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations)

genotype = neat.get_best_genotype()
print(genotype)
ann = Ann(genotype)

for item in dataset.get_dataset():
    result = ann.calculate(item.input)
    print(str(item) + "; Result " + str(result))
