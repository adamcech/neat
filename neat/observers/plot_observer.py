import os

from typing import List, Union, Tuple, Dict

import networkx as nx

from neat.encoding.node_type import NodeType
from neat.observers.abstract_observer import AbstractObserver

import matplotlib.pyplot as plt

import multiprocessing as mp
import numpy as np


class PlotObserver(AbstractObserver):

    def __init__(self, dir_path: str, plot_counter=10):
        self._generation = 0
        self._plot_counter = plot_counter

        self.dir_path = dir_path

        self._max_score = []  # type: List[float]
        self._min_score = []  # type: List[float]
        self._avg_score = []  # type: List[float]
        self._avg_score_std_plus = []  # type: List[float]
        self._avg_score_std_minus = []  # type: List[float]

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

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def end_generation(self, neat: "Neat") -> None:
        population = neat.population
        best = population.get_best_member()

        self._max_score.append(best.score)
        self._min_score.append(population.get_worst_member().score)
        self._avg_score.append(sum(p.score for p in population.population) / len(population.population))
        avg_score_std = np.std(self._avg_score)
        self._avg_score_std_plus.append(self._avg_score[-1] + avg_score_std)
        self._avg_score_std_minus.append(self._avg_score[-1] - avg_score_std)

        nodes_avg, nodes_max, nodes_min = self._get_nodes_info(population)

        self._nodes_max.append(nodes_max)
        self._nodes_min.append(nodes_min)
        self._nodes_avg.append(nodes_avg)
        nodes_avg_std = np.std(self._nodes_avg)
        self._nodes_avg_std_plus.append(self._nodes_avg[-1] + nodes_avg_std)
        self._nodes_avg_std_minus.append(self._nodes_avg[-1] - nodes_avg_std)

        edges_avg, edges_max, edges_min = self._get_edges_info(population)

        self._edges_max.append(edges_max)
        self._edges_min.append(edges_min)
        self._edges_avg.append(edges_avg)
        edges_avg_std = np.std(self._edges_avg)
        self._edges_avg_std_plus.append(self._edges_avg[-1] + edges_avg_std)
        self._edges_avg_std_minus.append(self._edges_avg[-1] - edges_avg_std)

        if self._generation % self._plot_counter == self._plot_counter - 1 and self._generation != 0:
            try:
                p = mp.Process(target=self.__plot, args=(best,))
                p.start()
                p.join()
            except Exception:
                print("ploting ex")

    def __plot(self, best: "Genotype"):
        self._plot_curves([self._max_score, self._avg_score],
                          ["max", "avg"],
                          ["r-", "b-"],
                          "Score",
                          self._get_base_path("score.svg"))

        self._plot_curves([self._nodes_max, self._nodes_min, self._nodes_avg],
                          ["Nodes Max", "Nodes Min", "Nodes Avg"],
                          ["r-", "k-", "b-"],
                          "Nodes",
                          self._get_base_path("nodes.svg"))

        self._plot_curves([self._edges_max, self._edges_min, self._edges_avg],
                          ["Edges Max", "Edges Min", "Edges Avg"],
                          ["r-", "k-", "b-"],
                          "Edges",
                          self._get_base_path("edges.svg"))

        self._plot_best(best, self._get_base_path("best.svg"))

    def _plot_curves(self, curves: List[List[Union[int, float]]], labels: List[str], markers: List[str], header: str, save_path=None):
        generation = [i for i in range(len(curves[0]))]
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

    @staticmethod
    def _plot_best(genotype: "Genotype", save_path: str):
        g = nx.MultiDiGraph()

        for node in genotype.nodes:
            if node.type == NodeType.INPUT:
                g.add_node(str(node.id))
            if node.type == NodeType.HIDDEN:
                g.add_node(str(node.id))
            if node.type == NodeType.OUTPUT:
                g.add_node(str(node.id))

        for edge in [edge for edge in genotype.edges if edge.enabled]:
            g.add_edge(str(edge.input), str(edge.output), weight=edge.weight)

        node_color = []

        for node in genotype.nodes:
            if node.type == NodeType.INPUT:
                node_color.append("green")
            if node.type == NodeType.HIDDEN:
                node_color.append("red")
            if node.type == NodeType.OUTPUT:
                node_color.append("blue")

        pos = nx.fruchterman_reingold_layout(g)
        x_min, y_min, x_max, y_max = 10, 10, -10, -10
        for p in pos:
            x = pos[p][0]
            y = pos[p][1]

            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

        input_nodes = len([node for node in genotype.nodes if node.is_input()])
        output_nodes = len([node for node in genotype.nodes if node.is_output()])
        x_size = abs(x_min) + abs(x_max)
        y_size = abs(y_min) + abs(y_max)
        ix = x_min - (x_size * 0.15)
        iy = y_max - (y_size * 0.1)
        iy_step = y_size / input_nodes
        ox = x_max + (x_size * 0.15)
        oy = y_max - (y_size * 0.1)
        oy_step = y_size / output_nodes
        for node in [node for node in genotype.nodes if node.is_input()]:
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ix
                    pos[p][1] = iy
                    iy -= iy_step
                    break

        for node in [node for node in genotype.nodes if node.is_output()]:
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ox
                    pos[p][1] = oy
                    oy -= oy_step
                    break

        nx.draw(g, pos=pos, node_size=50, with_labels=False, node_color=node_color, line_color='grey', linewidths=0, width=0.1)

        plt.savefig(save_path)
        plt.close()

    def _get_base_path(self, file_name: str):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        return self.dir_path + os.path.sep + str(self._generation + 1) + "_" + file_name

    def _get_nodes_info(self, population: "Population") -> Tuple[float, int, int]:
        avg = []
        maximum = 0
        minimum = 9999999

        for member in population.population:
            member_nodes = len([n for n in member.nodes if n.is_hidden() or n.is_output()])

            avg.append(member_nodes)

            if member_nodes > maximum:
                maximum = member_nodes

            if member_nodes < minimum:
                minimum = member_nodes

        return sum(avg) / len(avg), maximum, minimum

    def _get_edges_info(self, population: "Population") -> Tuple[float, int, int]:
        avg = []
        maximum = 0
        minimum = 9999999

        for member in population.population:
            member_edges = len([e for e in member.edges if e.enabled])

            avg.append(member_edges)

            if member_edges > maximum:
                maximum = member_edges

            if member_edges < minimum:
                minimum = member_edges

        return sum(avg) / len(avg), maximum, minimum
