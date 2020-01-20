import math
import random

import numpy as np
from typing import List, Tuple

from neat.encoding.edge import Edge
from neat.encoding.genotype import Genotype
from neat.encoding.node import Node


class Species:
    """Genotype Species"""

    __stagnation_time = 999999
    __extinct_size = 2
    __dying_size = 5

    __id_counter = 1

    def __init__(self, representative: Genotype):
        self.id = Species.__id_counter
        Species.__id_counter += 1

        self.representative = representative
        self.members = [self.representative]  # type: List[Genotype]
        self.compatibility = [1]  # type: List[float]

        self.score_history = []
        self.age = 0
        self.fitness = 0
        self.score = 0

        self._best_member = None

    def add_member(self, genotype: Genotype, compatibility: float):
        self.members.append(genotype)
        self.compatibility.append(compatibility)

    def reset(self):
        self._best_member = None
        self.score_history.append(max([genotype.score for genotype in self.members]))
        self.score_history = self.score_history[-self.__stagnation_time:]

        self.age += 1
        if self.age % 10 == 0:
            self.representative = random.sample(self.members, 1)[0]

        self.members = [self.representative]
        self.compatibility = [1]

    def union(self, other: "Species"):
        self._best_member = self.get_best_member() if self.get_best_member().score >= other.get_best_member().score else other.get_best_member()
        self.members += other.members
        self.compatibility += other.compatibility
        self.score_history = np.average(np.array([self.score_history, other.score_history]), axis=0)

    def get_best_member(self) -> Genotype:
        return self.members[np.argsort([member.score for member in self.members])[-1]]

    def evaluate_fitness(self):
        self.fitness = 0 if len(self.members) == 0 else sum(member.fitness for member in self.members) / len(self.members)
        self.score = 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def remove_worst(self, percentage: float):
        sort_indices = np.argsort([member.score for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(int(len(self.members) * (1 - percentage)))]

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def is_extinct(self) -> bool:
        return len(self.members) <= Species.__extinct_size

    def is_dying(self):
        return len(self.members) <= Species.__dying_size

    def is_mature(self):
        return len(self.score_history) == self.__stagnation_time

    def is_stagnating(self) -> bool:
        return max(self.score_history) == self.score_history[0] if self.is_mature() else False

    def get_size(self) -> int:
        return len(self.members)

    def __repr__(self):
        return "### Fitness: " + str(self.score) + ", Members: " + str(len(self.members)) + " ###"

    def get_elite(self, elitism: float) -> List[Genotype]:
        sort_indices = np.argsort([member.score for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(len(sort_indices))]

        return self.members[:math.ceil(len(self.members) * elitism)]

    def get_casualties(self, count: int):
        survivors = len(self.members) - count
        sort_indices = np.argsort([member.score for member in self.members])[::-1]

        casualties = [self.members[sort_indices[i]] for i in range(survivors, survivors + count)]

        self.compatibility = [self.compatibility[sort_indices[i]] for i in range(survivors)]
        self.members = [self.members[sort_indices[i]] for i in range(survivors)]

        return casualties

    def add_casualties(self, genotypes: List[Genotype]):
        species_genotypes_nodes = {}
        species_genotypes_edges = {}

        for i in range(len(genotypes)):
            template_genotype = self.members[i % len(self.members)]
            genotype = genotypes[i]

            template_genotype_nodes = species_genotypes_nodes.get(template_genotype)
            if template_genotype_nodes is None:
                template_genotype_nodes = {node.id: node.type for node in template_genotype.nodes}
                species_genotypes_nodes[template_genotype] = template_genotype_nodes

            template_genotype_edges = species_genotypes_edges.get(template_genotype)
            if template_genotype_edges is None:
                template_genotype_edges = {edge.innovation: (edge.input, edge.output, edge.enabled) for edge in template_genotype.edges}
                species_genotypes_edges[template_genotype] = template_genotype_edges

            genotype_edges = {edge.innovation: (edge.input, edge.output, edge.enabled, edge.weight) for edge in genotype.edges}

            genotype.nodes = [Node(node_id, template_genotype_nodes[node_id]) for node_id in template_genotype_nodes]
            genotype.edges = []
            for edge_innovation in template_genotype_edges:
                edge_input, edge_output, edge_enabled = template_genotype_edges[edge_innovation]
                orig_edge = genotype_edges.get(edge_innovation)
                genotype.edges.append(Edge(edge_input, edge_output, edge_enabled, edge_innovation, weight=orig_edge if orig_edge is None else orig_edge[3]))

            max_additions = math.floor(len(template_genotype_edges) * (1 + ((1 - Genotype.compatibility_threshold) * 0.35)))
            for edge_innovation in genotype_edges:
                edge_input, edge_output, edge_enabled, edge_weight = genotype_edges[edge_innovation]

                if edge_input in template_genotype_nodes and edge_output in template_genotype_nodes and edge_enabled and edge_innovation not in template_genotype_edges:
                    genotype.edges.append(Edge(edge_input, edge_output, edge_enabled, edge_innovation, weight=edge_weight))
                    max_additions -= 1

                if max_additions == 0:
                    break

            genotype.ancestor = None
            self.add_member(genotype, 1)

    def calculate_compatibility(self, genotype: Genotype) -> Tuple[bool, float]:
        return Genotype.is_compatible(genotype, self.representative)
