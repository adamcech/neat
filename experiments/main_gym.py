from gym_client.cart_pole_client import CartPoleClient
from gym_client.lunar_lander_client import LunarLanderClient
from neat.ann.ann import Ann
from neat.neat import Neat

c1 = 1.5
c2 = 1.5
c3 = 1.0
t = 1.5

population_size = 150
generations = 100000

dataset = LunarLanderClient()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations)

genotype = neat.get_best_genotype()
print(genotype)

dataset.render(Ann(genotype))
