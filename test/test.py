from gym_client.bipedal_walker_client import BipedalWalkerClient
from gym_client.cart_pole_client import CartPoleClient
from neat.ann.ann import Ann
from neat.encoding.genotype import Genotype
from neat.neat import Neat

genotype = Genotype.initial_genotype(CartPoleClient(), [10, 5, 3])

print(genotype)

print("end")