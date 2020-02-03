from neat.config import Config
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.plot_observer import PlotObserver
from neat.observers.render_observer import RenderObserver


def xor():
    config_xor = "./configs/xor.txt"
    neat = Neat(Config(config_xor))
    neat.run()


def bipedal_walker():
    config_bipedal_walker = "./configs/bipedal_walker.txt"

    neat = Neat(Config(config_bipedal_walker))

    neat.add_observer(PlotObserver(neat.config.dir_path, 50))
    neat.add_observer(ConsoleObserver())
    neat.add_observer(RenderObserver(50))
    neat.add_observer(AutosaveObserver(neat.config.dir_path, 50))

    neat.run()


def bipedal_walker_hardcore():
    config_bipedal_walker_hardcore = "./configs/bipedal_walker_hardcore.txt"

    neat = Neat(Config(config_bipedal_walker_hardcore))

    neat.add_observer(PlotObserver(neat.config.dir_path, 50))
    neat.add_observer(ConsoleObserver())
    neat.add_observer(RenderObserver(50))
    neat.add_observer(AutosaveObserver(neat.config.dir_path, 50))

    neat.run()


def main():
    xor()
    # bipedal_walker()
    # bipedal_walker_hardcore()


if __name__ == "__main__":
    main()
