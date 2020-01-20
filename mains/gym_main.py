from gym_client.acrobot_client import AcrobotClient
from gym_client.bipedal_walker_client import BipedalWalkerClient
from gym_client.bipedal_walker_hardcore_client import BipedalWalkerHardcoreClient
from gym_client.breakout_client import BreakoutClient
from gym_client.cart_pole_client import CartPoleClient
from gym_client.lunar_lander_client import LunarLanderClient
from gym_client.pacman_client import PacmanClient
from neat.config import Config
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.plot_observer import PlotObserver
from neat.observers.render_observer import RenderObserver


def main():
    dataset = BipedalWalkerHardcoreClient()

    config = Config(
        c1=1.0,
        c2=1.0,
        c3=0.4,
        t=3.0,
        population_size=150,
        train_counter=100,
        train_max_iterations=100,
        train_elitism=0.2,
        train_f=0.6,
        train_cr=0.9,
        input_size=dataset.get_input_size(),
        bias_size=dataset.get_bias_size(),
        output_size=dataset.get_output_size())

    neat = Neat(config, dataset)

    plot_observer = PlotObserver(50)
    neat.add_observer(plot_observer)
    neat.add_observer(ConsoleObserver())
    neat.add_observer(RenderObserver(dataset, 25))
    neat.add_observer(AutosaveObserver(plot_observer.dir_path, 50))

    neat.next_generations(10000000)


if __name__ == "__main__":
    main()
