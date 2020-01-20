import random

from neat.ann.ann import Ann
from neat.neat import Neat

neat = Neat.open("../../bipedal_hardcore/files_5/1000_autosave")
best = neat.population.get_best_member()

best_seeds = [5901242, 1864762, 5290009, 5861313, 1027648, 9002802, 8259407, 8170113, 5859637, 1284802]

random_seeds = random.sample(range(10000000), 10)

best_and_random_seeds = best_seeds + random_seeds
random.shuffle(best_and_random_seeds)

neat.dataset.render(Ann(best), best_seeds)
