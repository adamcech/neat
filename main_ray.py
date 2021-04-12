import os
import socket
import time
import pickle
import subprocess

from neat.config import Config
from neat.neat import Neat
from neat.observers.autosave_observer import AutosaveObserver
from neat.observers.console_observer import ConsoleObserver
from neat.observers.csv_observer import CsvObserver
from neat.observers.plot_observer import PlotObserver

import ray

print("start")
config_path = "./configs/ray.txt"


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
        # subprocess.call(["/home/cec0113/.local/bin/ray", "start", "--head", "--redis-port=31265"])

        ip = str(socket.gethostbyname(socket.gethostname())) + ":" + "6379"
        info = {"ip": ip}
        with open(config.ray_info_loc, 'wb') as file:
            pickle.dump(info, file)
        print("SERVER IP", info)

        if p_call:
            subprocess.call(["qsub", "-q", "qexp", "-l", "select=8:ncpus=24", "/home/cec0113/myjob_multi_ray"])
            ray.init(num_cpus=24)
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
        neat.start()
        neat_start(run + 1, p_call)
    else:
        time.sleep(10)
        with open(config.ray_info_loc, 'rb') as file:
            info = pickle.load(file)
        print("RAY NODE CONNECT", info)
        subprocess.call(["/home/cec0113/.local/bin/ray", "start", "--address="+info["ip"]])


if __name__ == "__main__":
    main()
