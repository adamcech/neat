import os
from typing import List, Union, Tuple, Dict

from neat.population import Population
from neat.observers.abstract_observer import AbstractObserver

import matplotlib.pyplot as plt
import neat_tools

import numpy as np


class PlotObserver(AbstractObserver):

    def __init__(self, plot_counter=10):
        self._generation = 0
        self._plot_counter = plot_counter

        self.dir_path = "/home/adam/Workspace/pycharm/neat/graphs_"
        dir_counter = 0

        while True:
            if os.path.exists(self.dir_path + str(dir_counter)):
                dir_counter += 1
            else:
                self.dir_path += str(dir_counter)
                os.makedirs(self.dir_path)
                break

        self._max_score = []  # type: List[float]
        self._avg_score = []  # type: List[float]
        self._avg_score_std_plus = []  # type: List[float]
        self._avg_score_std_minus = []  # type: List[float]

        self._max_fitness = []  # type: List[float]
        self._avg_fitness = []  # type: List[float]
        self._avg_fitness_std_plus = []  # type: List[float]
        self._avg_fitness_std_minus = []  # type: List[float]

        self._nodes_avg = []  # type: List[float]
        self._nodes_min = []  # type: List[int]
        self._nodes_max = []  # type: List[int]
        self._nodes_avg_std_plus = []  # type: List[float]
        self._nodes_avg_std_minus = []  # type: List[float]

        self._edges_avg = []  # type: List[float]
        self._edges_min = []  # type: List[int]
        self._edges_max = []  # type: List[int]
        self._edges_avg_std_plus = []  # type: List[float]
        self._edges_avg_std_minus = []  # type: List[float]

        self._species_pos = {}  # type: Dict[int, int]
        self._species_sizes = []  # type: List[List[Tuple[int, int]]]

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def end_generation(self, population: Population) -> None:
        best = population.get_best()

        self._max_score.append(best.score)
        self._avg_score.append(population.get_avg_score())
        avg_score_std = np.std(self._avg_score)
        self._avg_score_std_plus.append(self._avg_score[-1] + avg_score_std)
        self._avg_score_std_minus.append(self._avg_score[-1] - avg_score_std)

        self._max_fitness.append(best.fitness)
        self._avg_fitness.append(population.get_avg_fitness())
        avg_fitness_std = np.std(self._avg_fitness)
        self._avg_fitness_std_plus.append(self._avg_fitness[-1] + avg_fitness_std)
        self._avg_fitness_std_minus.append(self._avg_fitness[-1] - avg_fitness_std)

        nodes_avg, nodes_max, nodes_min = population.get_nodes_info()

        self._nodes_max.append(nodes_max)
        self._nodes_min.append(nodes_min)
        self._nodes_avg.append(nodes_avg)
        nodes_avg_std = np.std(self._nodes_avg)
        self._nodes_avg_std_plus.append(self._nodes_avg[-1] + nodes_avg_std)
        self._nodes_avg_std_minus.append(self._nodes_avg[-1] - nodes_avg_std)

        edges_avg, edges_max, edges_min = population.get_edges_info()

        self._edges_max.append(edges_max)
        self._edges_min.append(edges_min)
        self._edges_avg.append(edges_avg)
        edges_avg_std = np.std(self._edges_avg)
        self._edges_avg_std_plus.append(self._edges_avg[-1] + edges_avg_std)
        self._edges_avg_std_minus.append(self._edges_avg[-1] - edges_avg_std)

        curr_species = []
        for species in population.get_species():
            if self._species_pos.get(species.id) is None:
                self._species_pos[species.id] = len(self._species_pos)
            curr_species.append((species.id, len(species.members)))

        self._species_sizes.append(curr_species)

        if self._generation % self._plot_counter == 0 and self._generation != 0:
            self._plot_curves([self._max_score, self._avg_score, self._avg_score_std_plus, self._avg_score_std_minus], ["max", "avg", "+std", "-std"], ["r-", "b-", "g-.", "g-."], "Score", self._get_base_path("score.svg"))
            # self._plot_curves([self._max_fitness, self._avg_fitness, self._avg_fitness_std_plus, self._avg_fitness_std_minus], ["max", "avg", "+std", "-std"], ["r-", "b-", "g-.", "g-."], "Fitness", self._get_base_path("fitness.svg"))
            self._plot_curves([self._nodes_max, self._nodes_min, self._nodes_avg, self._nodes_avg_std_plus, self._nodes_avg_std_minus], ["max", "min", "avg", "+std", "-std"], ["r-", "k-", "b-", "g-.", "g-."], "Nodes", self._get_base_path("nodes.svg"))
            self._plot_curves([self._edges_max, self._edges_min, self._edges_avg, self._edges_avg_std_plus, self._edges_avg_std_minus], ["max", "min", "avg", "+std", "-std"], ["r-", "k-", "b-", "g-.", "g-."], "Edges", self._get_base_path("edges.svg"))
            self._plot_species(self._get_base_path("species.svg"))
            self._plot_best(best, self._get_base_path("best.svg"))

    def _plot_curves(self, curves: List[List[Union[int, float]]], labels: List[str], markers: List[str], header: str, save_path=None):
        generation = [i for i in range(self._generation + 1)]
        size = len(labels)

        for i in range(size):
            plt.plot(generation, curves[i], markers[i], label=labels[i])

        plt.title(header)
        plt.xlabel("Generations")
        plt.ylabel(header)
        plt.grid()
        plt.legend(loc="best")

        plt.savefig(save_path)
        plt.close()

    def _plot_species(self, save_path: str):
        species_sizes = []
        for generation in range(len(self._species_sizes)):
            sizes = self._species_sizes[generation]
            curr = {self._species_pos[species_id]: size for species_id, size in sizes}
            curr_list = []
            for i in range(len(self._species_pos)):
                curr_item = curr.get(i)
                curr_list.append(0 if curr_item is None else curr_item)
            species_sizes.append(curr_list)

        curves = np.array(species_sizes).T

        fig, ax = plt.subplots()
        ax.stackplot(range(self._generation + 1), *curves)

        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def _plot_best(best: "Genotype", save_path: str):
        neat_tools.visualize_genotype(best, save_path)

    def _get_base_path(self, file_name: str):
        return self.dir_path + "/" + str(self._generation) + "_" + file_name
