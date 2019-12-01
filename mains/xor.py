from dataset.dataset_xor import DatasetXor
from neat.ann.ann import Ann
from neat.neat import Neat
from neat.observers.console_observer import ConsoleObserver
from neat.observers.plot_observer import PlotObserver
from neat.observers.render_observer import RenderObserver

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3.0

population_size = 200
generations = 100000

dataset = DatasetXor()

neat = Neat(c1, c2, c3, t, population_size, dataset)

neat.add_observer(ConsoleObserver())
# neat.add_observer(PlotObserver())
neat.add_observer(RenderObserver(dataset))

neat.next_generations(generations)



"""
c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3

population_size = 150
generations = 2000

dataset = DatasetXor()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations)

genotype = neat.get_best_genotype()
print(genotype)
dataset.render(Ann(genotype))
"""
