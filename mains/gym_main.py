from gym_client.acrobot_client import AcrobotClient
from gym_client.bipedal_walker_client import BipedalWalkerClient
from gym_client.bipedal_walker_hardcore_client import BipedalWalkerHardcoreClient
from gym_client.cart_pole_client import CartPoleClient
from gym_client.lunar_lander_client import LunarLanderClient
from neat.ann.ann import Ann
from neat.neat import Neat

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3.0

population_size = 200
generations = 100000

dataset = LunarLanderClient()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations)

genotype = neat.get_best_genotype()
print(genotype)

dataset.render(Ann(genotype))
