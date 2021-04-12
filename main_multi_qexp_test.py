import os
import socket
import time

import ray

from evaluators.cluster.cluster_node import ClusterNode
from neat.config import Config
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.csv_observer import CsvObserver
from neat.observers.plot_observer import PlotObserver


import numpy as np

from neat.observers.render_observer import RenderObserver

print("loaded")
config_path = "./configs/iris.txt"


def main():
    print("main")
    config = Config(config_path)
    # config.tcp_port += np.random.randint(0, 1000)
    config.cluster_evaluation = False
    # config.cluster_nodes_loc = "/home/adam/Workspace/pycharm/neat/nodes/"
    # config.cluster_main_loc = "/home/adam/Workspace/pycharm/neat/nodes/"
    # config.work_dir = "/home/adam/Results/local_0/"
    # config.ray_info_loc = "/home/cec0113/ray_info/info"

    # config.cluster_evaluation = True
    # ray.init()

    if os.path.isfile(config.work_dir + "autosave"):
        config = Neat.open(config.work_dir + "autosave").config

    print("Starting neat")
    if os.path.isfile(config.work_dir + "autosave"):
        print("opening")
        neat = Neat.open(config.work_dir + "autosave")
    else:
        print("initing")
        neat = Neat(config)
        neat.add_observer(CsvObserver(neat.config))
        neat.add_observer(ConsoleObserver(neat.config.work_dir, neat.config))
        neat.add_observer(PlotObserver(neat.config.work_dir, 25))
        neat.add_observer(RenderObserver(25))
        # neat.add_observer(AutosaveObserver(neat.config.work_dir, neat.config.walltime_sec, 25))

    print("start")
    print(neat.config.dataset_name)
    neat.start()


if __name__ == "__main__":
    main()
