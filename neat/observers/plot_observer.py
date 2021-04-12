import os

from typing import List, Union, Tuple, Set, Dict

import networkx as nx

from neat.encoding.genotype import Genotype
from neat.encoding.node import Node
from neat.encoding.node_activation import NodeActivation
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

        if (self._generation % self._plot_counter == self._plot_counter - 1 and self._generation != 0) or neat.config.done:
            try:
                p = mp.Process(target=self.__plot, args=(neat.config, "best_"+str(self._generation)+".svg", best, neat.max_score_history_gens, neat.max_score_history, neat.avg_score_history, neat.min_score_history))
                p.start()
                p.join()
            except Exception:
                print("ploting ex")

            try:
                p = mp.Process(target=self.__plot, args=(neat.config, "test_best_"+str(self._generation)+".svg", neat.best_genotype if neat.best_genotype is not None else best, neat.max_score_history_gens, neat.max_score_history, neat.avg_score_history, neat.min_score_history))
                p.start()
                p.join()
            except Exception:
                print("ploting ex")

    def __plot(self, config, file_name: str, best: "Genotype", max_score_gens: List[int], max_score: List[float], avg_score: List[float], min_score: List[float]):
        self._plot_curves([max_score, avg_score, min_score],
                          ["max", "avg", "min"],
                          ["r-", "k-", "b-"],
                          "Test",
                          self._get_base_path("test.svg"),
                          generations=max_score_gens)

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

        self._plot_best(config, best, self._get_base_path(file_name))

    def _plot_curves(self, curves: List[List[Union[int, float]]], labels: List[str], markers: List[str], header: str, save_path, **kwargs):
        size = len(labels)

        generation = kwargs.get("generations", None)
        if generation is None:
            generation = [i for i in range(len(curves[0]))]

        for i in range(size):
            plt.plot(generation, curves[i], markers[i], label=labels[i])

        plt.title(header)
        plt.xlabel("Generations")
        plt.ylabel(header)
        plt.grid()
        plt.legend(loc="best")

        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def _plot_best(config, genotype: "Genotype", save_path: str):
        g = nx.MultiDiGraph()

        for node in genotype.nodes:
            g.add_node(str(node.id))

        colors = []
        for e in [e for e in genotype.edges if e.enabled]:
            colors.append(e.weight / abs(config.min_weight) if e.weight <= 0 else abs(e.weight / config.max_weight))
            g.add_edge(str(e.input), str(e.output), weight=e.weight)

        node_color = []

        for node in genotype.nodes:
            if node.is_input() or node.is_bias():
                node_color.append("darkgreen")
            elif node.is_output():
                node_color.append("navy")

            elif node.activation == NodeActivation.SIN:
                node_color.append("purple")

            elif node.activation == NodeActivation.COS:
                node_color.append("pink")

            elif node.activation == NodeActivation.CLAMPED:
                node_color.append("red")

            elif node.activation == NodeActivation.TANH or node.activation == NodeActivation.STEEPENED_TANH:
                node_color.append("orange")

            elif node.activation == NodeActivation.RELU:
                node_color.append("yellow")

            elif node.activation == NodeActivation.SIGM or node.activation == NodeActivation.STEEPENED_SIGM:
                node_color.append("brown")

            elif node.activation == NodeActivation.ABS:
                node_color.append("grey")

            elif node.activation == NodeActivation.INV:
                node_color.append("black")

            elif node.activation == NodeActivation.GAUSS:
                node_color.append("cyan")

            elif node.activation == NodeActivation.LIN:
                node_color.append("lime")

            elif node.activation == NodeActivation.STEP:
                node_color.append("blue")

            elif node.activation == NodeActivation.SIGNUM:
                node_color.append("gainsboro")

        pos = nx.fruchterman_reingold_layout(g)
        x_min, y_min, x_max, y_max = 999999, 999999, -999999, -999999
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

        input_nodes = len([node for node in genotype.nodes if node.is_input() or node.is_bias()])
        output_nodes = len([node for node in genotype.nodes if node.is_output()])

        x_size = abs(x_min) + abs(x_max)
        y_size = abs(y_min) + abs(y_max)

        ix = x_min
        iy_step = y_size / (input_nodes + 1)
        iy = y_max - iy_step

        for node in [node for node in genotype.nodes if node.is_input() or node.is_bias()]:
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ix
                    pos[p][1] = iy
                    iy -= iy_step
                    break

        layers = genotype.get_node_layers()[1]
        layers_size = len(layers) + 2
        layer_step = x_size / layers_size

        for i in range(len(layers)):
            layer = layers[i]

            lx = x_min + (i + 1) * layer_step

            ly_step = y_size / (len(layer) + 1)
            ly = y_max - ly_step

            for node_id in layer:
                for p in pos:
                    if str(p) == str(node_id):
                        pos[p][0] = lx
                        pos[p][1] = ly
                        ly -= ly_step
                        break

        ox = x_min + (len(layers) + 1) * layer_step
        oy_step = y_size / (output_nodes + 1)
        oy = y_max - oy_step

        # for node in [node for node in genotype.nodes if node.is_output()]:
        #     for p in pos:
        #         if str(p) == str(node.id):
        #             pos[p][0] = ox
        #             pos[p][1] = oy
        #             oy -= oy_step
        #             break

        nodes_dict = {n.id: n for n in genotype.nodes}
        for node_id in layers[-1]:
            node = nodes_dict[node_id]
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ox
                    pos[p][1] = oy
                    oy -= oy_step
                    break

        nx.draw(g, pos, node_size=60, edge_color=colors, edge_cmap=plt.cm.coolwarm_r, node_color=node_color, width=0.2)

        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
        plt.close()

    """
    @staticmethod
    def _plot_best(genotype: "Genotype", save_path: str):
        g = nx.MultiDiGraph()

        for node in genotype.nodes:
            g.add_node(str(node.id))

        colors = []
        for e in [edge for edge in genotype.edges if edge.enabled]:
            colors.append(e.weight)
            g.add_edge(str(e.input), str(e.output), weight=e.weight)

        node_color = []

        for node in genotype.nodes:
            if node.type == NodeType.INPUT:
                node_color.append("green")
            if node.type == NodeType.HIDDEN:
                node_color.append("red")
            if node.type == NodeType.OUTPUT:
                node_color.append("blue")

        pos = nx.fruchterman_reingold_layout(g)
        x_min, y_min, x_max, y_max = 999999, 999999, -999999, -999999
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

        input_nodes = len([node for node in genotype.nodes if node.is_input() or node.is_bias()])
        output_nodes = len([node for node in genotype.nodes if node.is_output()])

        x_size = abs(x_min) + abs(x_max)
        y_size = abs(y_min) + abs(y_max)

        ix = x_min
        iy_step = y_size / (input_nodes + 1)
        iy = y_max - iy_step

        for node in [node for node in genotype.nodes if node.is_input() or node.is_bias()]:
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ix
                    pos[p][1] = iy
                    iy -= iy_step
                    break

        layers = PlotObserver.get_genotype_layers(genotype)
        layers_size = len(layers) + 2
        layer_step = x_size / layers_size

        for i in range(len(layers)):
            layer = layers[i]

            lx = x_min + (i + 1) * layer_step

            ly_step = y_size / (len(layer) + 1)
            ly = y_max - ly_step

            for node_id in layer:
                for p in pos:
                    if str(p) == str(node_id):
                        pos[p][0] = lx
                        pos[p][1] = ly
                        ly -= ly_step
                        break

        ox = x_min + (len(layers) + 1) * layer_step
        oy_step = y_size / (output_nodes + 1)
        oy = y_max - oy_step

        for node in [node for node in genotype.nodes if node.is_output()]:
            for p in pos:
                if str(p) == str(node.id):
                    pos[p][0] = ox
                    pos[p][1] = oy
                    oy -= oy_step
                    break

        nx.draw(g, pos, node_size=60, edge_color=colors, edge_cmap=plt.cm.coolwarm_r, node_color=node_color, width=0.2)

        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
        plt.close()
    """

    """
    @staticmethod
    def get_genotype_layers(genotype: Genotype) -> List[Set[int]]:
        layers = []
        nodes = {n.id: n for n in genotype.nodes}  # type: Dict[int, Node]
        layer = {n for n in genotype.nodes if n.is_output()}

        while True:
            next_layer = set()
            curr_id = {n.id for n in layer}

            for e in genotype.edges:
                input_neuron = nodes[e.input]
                if e.enabled and e.output in curr_id and input_neuron.is_hidden():
                    next_layer.add(input_neuron)

            if len(next_layer) > 0:
                layer = next_layer
                layers.append(next_layer)
            else:
                break

        layers = layers[::-1]

        for i in range(len(layers)):
            for n in layers[i]:
                for j in range(i + 1, len(layers)):
                    if n in layers[j]:
                        layers[j].remove(n)

        return [{n.id for n in layer} for layer in layers]
    """

    @staticmethod
    def get_genotype_layers(genotype: Genotype) -> List[Set[int]]:
        layers = {}  # type: Dict[int, Set[int]]

        for node in genotype.nodes:
            if node.is_hidden():
                node_id, node_layer = node.id, node.layer

                layer = layers.get(node_layer)
                if layer is None:
                    layers[node_layer] = {node_id}
                else:
                    layer.add(node_id)

        layers_id = list(layers)
        sorted_layers = np.argsort(layers_id)

        return [layers[layers_id[sorted_layers[i]]] for i in range(len(sorted_layers))]

    @staticmethod
    def get_genotype_layers_from_start(genotype: Genotype) -> List[Set[int]]:
        node_layers = {}  # type: Dict[int, int]
        layers_lists = {}  # type: Dict[int, Set[int]]

        node_types = {n.id: n for n in genotype.nodes}  # type: Dict[int, Node]
        node_outputs = {n.id: [] for n in genotype.nodes}  # type: Dict[int, List]

        for e in genotype.edges:
            if e.enabled:
                node_outputs[e.input].append(e.output)

        # curr_layer = min(n.layer for n in genotype.nodes if n.is_hidden())
        curr_layer = 1
        first_layer = [n.id for n in genotype.nodes if n.is_input()]

        while len(first_layer) >= 1:
            new_first_layer = []

            for node_id in first_layer:
                for input_id in node_outputs.get(node_id):
                    if node_types.get(input_id).is_output():
                        continue

                    new_first_layer.append(input_id)
                    ls = node_layers.get(input_id)

                    if ls is not None:
                        layers_lists.get(ls).remove(input_id)
                    node_layers[input_id] = curr_layer

                    layer = layers_lists.get(curr_layer)
                    if layer is None:
                        layers_lists[curr_layer] = {input_id}
                    else:
                        layer.add(input_id)

            curr_layer += 1
            first_layer = new_first_layer

        layers_id = list(layers_lists)
        sorted_layers = np.argsort(layers_id)

        return [layers_lists[layers_id[sorted_layers[i]]] for i in sorted_layers]

    def _get_base_path(self, file_name: str):
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        # return self.dir_path + os.path.sep + str(self._generation + 1) + "_" + file_name
        return self.dir_path + file_name

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
