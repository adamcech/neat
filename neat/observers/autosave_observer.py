import pickle

from neat.observers.abstract_observer import AbstractObserver


class AutosaveObserver(AbstractObserver):

    def __init__(self, dir_path: str, autosave_generation=25):
        self.generation = 0
        self.dir_path = dir_path
        self.autosave_generation = autosave_generation

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def end_generation(self, neat: "Neat") -> None:
        if self.generation % self.autosave_generation == 0 and self.generation != 0:
            with open(self.dir_path + "/" + str(self.generation) + "_autosave", 'wb') as file:
                pickle.dump(neat, file)
