import os
import pickle
import time
from typing import List

from neat.observers.abstract_observer import AbstractObserver


class AutosaveObserver(AbstractObserver):

    def __init__(self, dir_path: str, walltime: int, autosave_generation=25):
        self.generation = 0
        self.dir_path = dir_path
        self.autosave_generation = autosave_generation

        self.start = 0
        self.walltime = 0
        self.init_walltime(walltime)

    def init_walltime(self, walltime: int):
        self.start = time.time()
        self.walltime = walltime

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def end_generation(self, neat: "Neat") -> None:
        is_save_gen = self.generation % self.autosave_generation == self.autosave_generation - 1 and self.generation != 0
        is_small_walltime = time.time() - self.start + 120 > self.walltime

        if is_save_gen or is_small_walltime or neat.config.done:
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path)

            if os.path.isfile(self.dir_path + "autosave"):
                if os.path.isfile(self.dir_path + "autosave_prev"):
                    os.remove(self.dir_path + "autosave_prev")
                os.rename(self.dir_path + "autosave", self.dir_path + "autosave_prev")

            with open(self.dir_path + "autosave", 'wb') as file:
                pickle.dump(neat, file)
