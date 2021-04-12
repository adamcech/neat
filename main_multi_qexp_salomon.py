import os
import socket
import time
import subprocess

from evaluators.cluster.cluster_node import ClusterNode
from neat.config import Config
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.csv_observer import CsvObserver
from neat.observers.plot_observer import PlotObserver


print("start")
config_path = "./configs/bipedal_walker_hardcore_salomon.txt"


def main():
    neat_start(0, True)


def neat_start(run: int, p_call: bool):
    print("main")
    config = Config(config_path)
    config.work_dir = config.work_dir[:-1] + "_" + str(run) + "/"

    neat = None
    try:
        if os.path.isfile(config.work_dir + "autosave"):
            neat = Neat.open(config.work_dir + "autosave")
    except EOFError:
        if os.path.isfile(config.work_dir + "autosave_prev"):
            neat = Neat.open(config.work_dir + "autosave_prev")

    if neat is not None:
        config = neat.config
        if config.done:
            neat_start(run + 1, p_call)

    files = os.listdir(config.cluster_main_loc)
    hostname = socket.gethostname()

    if any(hostname in h for h in files):
        if p_call:
            if config.cluster_evaluation:
                print("Starting node client")
                config.mp_max_proc = config.cluster_main_max_load
                ClusterNode(config, True)

                time.sleep(5)  # Wait for other cluster nodes to startup...
                subprocess.call(["qsub", "-q", "qexp", "-l", "select=8:ncpus=24", "/home/cec0113/myjob_multi"])
            else:
                subprocess.call(["qsub", "-q", "qexp", "-l", "select=1:ncpus=24", "/home/cec0113/myjob_multi"])

        p_call = False

        print("Starting neat")
        if neat is None:
            print("initing")
            neat = Neat(config)
            neat.add_observer(CsvObserver(neat.config))
            neat.add_observer(ConsoleObserver(neat.config.work_dir, neat.config))
            neat.add_observer(PlotObserver(neat.config.work_dir, 25))
            neat.add_observer(AutosaveObserver(neat.config.work_dir, neat.config.walltime_sec, 25))

        print("run")
        print("Start")
        neat.start()
        neat_start(run + 1, p_call)
    else:
        print("Starting node")
        ClusterNode(config).run()


if __name__ == "__main__":
    main()
