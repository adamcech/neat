from gym_client.asteroids_client import AsteroidsClient
from gym_client.breakout_client import BreakoutClient
from gym_client.pacman_client import PacmanClient
from gym_client.pole_balancing_client import PoleBalancingClient
from gym_client.pong_client import PongClient
from gym_client.space_invaders_client import SpaceInvadersClient
from neat.ann.ann import Ann
from neat.neat import Neat

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3

population_size = 150
generations = 100000

# dataset = AsteroidsClient()
# dataset = BreakoutClient()
# dataset = PacmanClient()
# dataset = PongClient()
dataset = SpaceInvadersClient()
# dataset = PoleBalancingClient()

neat = Neat(c1, c2, c3, t, population_size, dataset)
neat.next_generations(generations, output=True, render=True)

genotype = neat.get_best_genotype()
print(genotype)

dataset.render(Ann(genotype))
