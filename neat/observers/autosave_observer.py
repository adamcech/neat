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
        if self.generation % self.autosave_generation == self.autosave_generation - 1 and self.generation != 0:
            neat.population.remove_recursions()
            with open(self.dir_path + "/" + str(self.generation + 1) + "_autosave", 'wb') as file:
                pickle.dump(neat, file)
