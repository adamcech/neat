from gym_client.pole_balancing_client import PoleBalancingClient
from neat.ann.ann import Ann
from neat.neat import Neat
from neat.population import Population

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3

population_size = 150
generations = 1

dataset = PoleBalancingClient()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations, output=True)

genotype = neat.get_best_genotype()
print(genotype)

print(str(Population.eval_time))

dataset.render(Ann(genotype))
