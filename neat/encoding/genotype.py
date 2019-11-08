import copy
import random
from typing import List

import numpy as np

from dataset.dataset import Dataset
from neat.encoding.innovation_map import InnovationMap
from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType


class Genotype:
    """Holds both nodes and connection
    """

    compatibility_threshold = 0.9
    __innovation_map = InnovationMap()
    __f = 0.1
    __cr = 0.85

    def __init__(self):
        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]
        self.fitness = 0
        self.score = 0

    @staticmethod
    def initial_genotype(dataset: Dataset) -> "Genotype":
        genotype = Genotype()

        # Adding input nodes
        for i in range(dataset.get_input_size() + dataset.get_bias_size()):
            genotype.nodes.append(Node(Genotype.__innovation_map.next_innovation(), NodeType.INPUT))

        # Adding output nodes
        for i in range(dataset.get_output_size()):
            genotype.nodes.append(Node(Genotype.__innovation_map.next_innovation(), NodeType.OUTPUT))

        # Connecting inputs to outputs
        for input_node in [node for node in genotype.nodes if node.is_input()]:
            for output_node in [node for node in genotype.nodes if node.is_output()]:
                genotype.edges.append(
                    Edge(input_node.id, output_node.id, True,
                         Genotype.__innovation_map.get_edge_innovation(input_node.id, output_node.id)))

        """
        # Connectiong only bias to outputs
        for i in range(dataset.get_input_size(), dataset.get_input_size() + dataset.get_bias_size()):
            bias_node = genotype.nodes[i]
            for output_node in [node for node in genotype.nodes if node.is_output()]:
                genotype.edges.append(
                    Edge(bias_node.id, output_node.id, True,
                         Genotype.__innovation_map.get_edge_innovation(bias_node.id, output_node.id)))
        """

        return genotype

    def initial_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for connection in self.edges:
            connection.mutate_random_weight()
        return cp

    def deepcopy(self) -> "Genotype":
        return copy.deepcopy(self)

    def mutate_add_node(self, dataset: Dataset):
        biases_id = [i for i in range(dataset.get_input_size(), dataset.get_input_size() + dataset.get_bias_size())]

        for i in range(50):
            old_edge = random.sample(self.edges, 1)[0]  # type: Edge
            if old_edge.input in biases_id:
                continue

            old_edge.enabled = False

            new_node = Node(Genotype.__innovation_map.get_node_innovation(old_edge.input, old_edge.output, self.nodes), NodeType.HIDDEN)
            new_edge = Edge(old_edge.input, new_node.id, True, Genotype.__innovation_map.get_edge_innovation(old_edge.input, new_node.id), weight=1)
            original_edge = Edge(new_node.id, old_edge.output, True, Genotype.__innovation_map.get_edge_innovation(new_node.id, old_edge.output), weight=old_edge.weight)

            self.nodes.append(new_node)
            self.edges.append(new_edge)
            self.edges.append(original_edge)
            break

    def mutate_add_edge(self):
        for try_counter in range(0, 10):
            samples = random.sample(self.nodes, 2)
            input_node = samples[0]  # type: Node
            output_node = samples[1]  # type: Node
            if (input_node.type == NodeType.OUTPUT or output_node.type == NodeType.INPUT) or \
               (input_node.type == NodeType.INPUT and output_node.type == NodeType.INPUT) or \
               (input_node.type == NodeType.OUTPUT and output_node.type == NodeType.OUTPUT) or \
               any((edge.input == input_node.id and edge.output == output_node.id) or
                   (edge.input == output_node.id and edge.output == input_node.id) for edge in self.edges):
                continue

            if not self._is_new_edge_recurrent(input_node.id, output_node.id):
                self.edges.append(
                    Edge(input_node.id,
                         output_node.id,
                         True,
                         Genotype.__innovation_map.get_edge_innovation(input_node.id, output_node.id)))
                break

    def calculate_fitness(self, dataset: Dataset):
        self.fitness = dataset.get_fitness(Ann(self))
        self.score = self.fitness

    def _is_new_edge_recurrent(self, input_node: int, output_node: int) -> bool:
        # Search for path from output_node to input_node
        # TODO better solution
        visited = []
        self._dfs(output_node, visited)
        return input_node in visited

    def _dfs(self, start_node: int, visited: List[int]):
        visited.append(start_node)

        for node in self._get_adjacent_nodes(start_node):
            if node not in visited:
                self._dfs(node, visited)

    def _get_adjacent_nodes(self, node_id: int) -> List[int]:
        nodes = []
        for edge in self.edges:
            if edge.input == node_id and edge.output not in nodes:
                nodes.append(edge.output)
        return nodes

    def __str__(self):
        return str(self.score) + "\n" + \
               str(len(self.nodes)) + " " + str(self.nodes) + "\n" + \
               str(sum([1 for edge in self.edges if edge.enabled])) + " " + str(self.edges)

    @staticmethod
    def crossover(mom: "Genotype", dad: "Genotype") -> "Genotype":
        better = mom if mom.fitness > dad.fitness else dad
        worse = mom if mom.fitness <= dad.fitness else dad

        genotype = Genotype()

        for node in better.nodes:
            genotype.nodes.append(Node(node.id, node.type))

        better_counter, worse_counter = 0, 0

        while better_counter < len(better.edges) and worse_counter < len(worse.edges):
            skip = False
            b = better.edges[better_counter]
            w = worse.edges[worse_counter]
            selected = b

            if b.innovation == w.innovation:
                better_counter += 1
                worse_counter += 1
                if np.random.uniform(0, 1) < 0.5:
                    selected = w
            elif b.innovation < w.innovation:
                better_counter += 1
            elif b.innovation > w.innovation:
                worse_counter += 1
                skip = True

            if not skip:
                genotype._copy_edge_from_crossover(selected)

        for i in range(better_counter, len(better.edges)):
            genotype._copy_edge_from_crossover(better.edges[i])

        return genotype

    def mark_ancestor(self, ancestor: "Genotype") -> None:
        self_counter, ancestor_counter = 0, 0

        while self_counter < len(self.edges) and ancestor_counter < len(ancestor.edges):
            s = self.edges[self_counter]
            a = ancestor.edges[ancestor_counter]

            if s.innovation == a.innovation:
                self_counter += 1
                ancestor_counter += 1
                if np.random.uniform() > self.__cr:
                    s.weight = a.weight
                    s.enabled = a.enabled
                    s.mutable = False
            elif s.innovation < a.innovation:
                self_counter += 1
            elif s.innovation > a.innovation:
                ancestor_counter += 1

    def _copy_edge_from_crossover(self, edge: Edge):
        self.edges.append(Edge(edge.input, edge.output, edge.enabled, edge.innovation, weight=edge.weight))

    def _copy_edge_from_triple_crossover(self, best: Edge, parent0: Edge, parent1: Edge, rand_order: List[int]):
        if best.enabled:
            vals = [best.weight, parent0.weight, parent1.weight]
            weight = vals[rand_order[0]] + self.__f * (vals[rand_order[1]] - vals[rand_order[2]]) if best.enabled else best.weight
            self.edges.append(Edge(best.input, best.output, best.enabled, best.innovation, weight=weight, mutable=False))
        else:
            self.edges.append(Edge(best.input, best.output, best.enabled, best.innovation, weight=best.weight, mutable=False))

    def _copy_edge_from_crossover_mutate(self, edge: Edge):
        self.edges.append(Edge(edge.input, edge.output, edge.enabled, edge.innovation, weight=edge.weight, mutable=True))

    @staticmethod
    def triple_crossover(p0: "Genotype", p1: "Genotype", p2: "Genotype") -> "Genotype":
        # Select best
        parents = [p0, p1, p2]
        parents_fitness = [p0.fitness, p1.fitness, p2.fitness]

        best = parents[parents_fitness.index(max(parents_fitness))]
        parents.remove(best)
        parent0, parent1 = parents[0], parents[1]

        """
        # Random
        best = p0
        parent0 = p1
        parent1 = p2
        """
        rand_order = random.sample(range(3), 3) if np.random.uniform() < 0.05 else [0, 1, 2]
        # rand_order = [0, 1, 2]

        genotype = Genotype()

        for node in best.nodes:
            genotype.nodes.append(Node(node.id, node.type))

        best_counter, parent0_counter, parent1_counter = 0, 0, 0

        while best_counter < len(best.edges) and parent0_counter < len(parent0.edges) and parent1_counter < len(parent1.edges):
            b = best.edges[best_counter]
            p0 = parent0.edges[parent0_counter]
            p1 = parent1.edges[parent1_counter]

            if b.innovation == p0.innovation == p1.innovation:
                genotype._copy_edge_from_triple_crossover(b, p0, p1, rand_order)
                best_counter += 1
                parent0_counter += 1
                parent1_counter += 1
            elif b.innovation > p0.innovation:
                parent0_counter += 1
            elif b.innovation > p1.innovation:
                parent1_counter += 1
            elif b.innovation == p0.innovation:
                selected = b if np.random.uniform() < 0.5 else p0
                genotype._copy_edge_from_crossover_mutate(selected)
                best_counter += 1
                parent0_counter += 1
            elif b.innovation == p1.innovation:
                selected = b if np.random.uniform() < 0.5 else p1
                genotype._copy_edge_from_crossover_mutate(selected)
                best_counter += 1
                parent1_counter += 1
            elif b.innovation < p0.innovation:
                genotype._copy_edge_from_crossover_mutate(b)
                best_counter += 1
            elif b.innovation < p1.innovation:
                genotype._copy_edge_from_crossover_mutate(b)
                best_counter += 1

        while best_counter < len(best.edges) and parent0_counter < len(parent0.edges):
            b = best.edges[best_counter]
            p0 = parent0.edges[parent0_counter]

            if b.innovation == p0.innovation:
                selected = b if np.random.uniform() < 0.5 else p0
                genotype._copy_edge_from_crossover_mutate(selected)
                best_counter += 1
                parent0_counter += 1
            elif b.innovation < p0.innovation:
                genotype._copy_edge_from_crossover_mutate(b)
                best_counter += 1
            elif b.innovation > p0.innovation:
                parent0_counter += 1

        while best_counter < len(best.edges) and parent1_counter < len(parent1.edges):
            b = best.edges[best_counter]
            p1 = parent1.edges[parent1_counter]

            if b.innovation == p1.innovation:
                selected = b if np.random.uniform() < 0.5 else p1
                genotype._copy_edge_from_crossover_mutate(selected)
                best_counter += 1
                parent1_counter += 1
            elif b.innovation < p1.innovation:
                genotype._copy_edge_from_crossover_mutate(b)
                best_counter += 1
            elif b.innovation > p1.innovation:
                parent1_counter += 1

        for i in range(best_counter, len(best.edges)):
            genotype._copy_edge_from_crossover_mutate(best.edges[i])

        return genotype

    def print_innovations(self):
        s = ""
        for edge in self.edges:
            s += "\t" + str(edge.innovation)
        print(s)

    @staticmethod
    def is_compatible(neat: "Neat", g0: "Genotype", g1: "Genotype"):
        if g0.edges[-1].innovation < g1.edges[-1].innovation:
            g0, g1 = g1, g0

        g0_n = len(g0.edges)
        g1_n = len(g1.edges)
        n = 1 if g0_n < 20 and g1_n < 20 else max(g0_n, g1_n)  # Number of genes in larger genotype, 1 if small genotype

        m = 0  # Matching genes
        d = 0  # Disjoint genes
        w = 0.0  # Weight difference of matching genes

        g0_counter, g1_counter = 0, 0

        while g0_counter < len(g0.edges) and g1_counter < len(g1.edges):
            g0_edge = g0.edges[g0_counter]
            g1_edge = g1.edges[g1_counter]

            if g0_edge.innovation == g1_edge.innovation:
                g0_counter += 1
                g1_counter += 1
                m += 1
                w += abs(g0_edge.weight - g1_edge.weight)
            elif g0_edge.innovation < g1_edge.innovation:
                g0_counter += 1
                d += 1
            elif g0_edge.innovation > g1_edge.innovation:
                g1_counter += 1
                d += 1

        e = len(g0.edges) - g0_counter  # Excess genes

        if g0_n > g1_n:
            g1_n, g0_n = g0_n, g1_n

        return (m / g1_n >= Genotype.compatibility_threshold), m / g1_n
        # return neat.t > neat.c1 * e / n + neat.c2 * d / n + neat.c3 * w / m
        # return neat.t > neat.c1 * e + neat.c2 * d + neat.c3 * w / m

    def __repr__(self):
        return str(self.fitness)
