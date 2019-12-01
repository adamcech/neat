from gym_client.acrobot_client import AcrobotClient
from gym_client.bipedal_walker_client import BipedalWalkerClient
from gym_client.bipedal_walker_hardcore_client import BipedalWalkerHardcoreClient
from gym_client.breakout_client import BreakoutClient
from gym_client.cart_pole_client import CartPoleClient
from gym_client.lunar_lander_client import LunarLanderClient
from gym_client.pacman_client import PacmanClient
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.plot_observer import PlotObserver
from neat.observers.render_observer import RenderObserver


def main():
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4
    t = 3.0

    population_size = 200
    generations = 100000

    dataset = LunarLanderClient()

    neat = Neat(c1, c2, c3, t, population_size, dataset)

    plot_observer = PlotObserver(10)
    neat.add_observer(plot_observer)
    neat.add_observer(ConsoleObserver())
    neat.add_observer(RenderObserver(dataset, 25))
    neat.add_observer(AutosaveObserver(plot_observer.dir_path, 25))

    neat.next_generations(generations)


if __name__ == "__main__":
    main()
