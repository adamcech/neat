from dataset.dataset_xor import DatasetXor
from neat.ann.ann import Ann
from neat.neat import Neat

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3.0
population_size = 150
generations = 100

dataset = DatasetXor()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations)

genotype = neat.get_best_genotype()
print(genotype)
ann = Ann(genotype)
for i in range(dataset.get_dataset_size()):
    dataset_item = dataset.next_item()
    result = ann.calculate(dataset_item.input)
    print(str(dataset_item) + "; Result " + str(result))
