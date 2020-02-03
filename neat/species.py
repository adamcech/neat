import math
import random

import numpy as np
from typing import List, Tuple

from neat.config import Config
from neat.encoding.edge import Edge
from neat.encoding.genotype import Genotype
from neat.encoding.node import Node


class Species:
    """Genotype Species"""

    def __init__(self, representative: Genotype, species_id: int, representative_change: int):
        self.id = species_id
        self.representative_change = representative_change

        self.representative = representative
        self.members = [representative]  # type: List[Genotype]
        self.compatibility = [1]  # type: List[float]

        self.age = 0
        self.fitness = 0
        self.score = 0

        self._best_member = None

    def add_member(self, genotype: Genotype, compatibility: float):
        self.members.append(genotype)
        self.compatibility.append(compatibility)

    def reset(self):
        self._best_member = None

        self.age += 1
        if self.age % self.representative_change == 0:
            self.representative = random.sample(self.members, 1)[0]

        self.members = [self.representative]
        self.compatibility = [1]

    def evaluate_fitness(self):
        self.fitness = 0 if len(self.members) == 0 else sum(member.fitness for member in self.members) / len(self.members)
        self.score = 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def remove_worst(self, percentage: float):
        sort_indices = np.argsort([member.score for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(int(len(self.members) * (1 - percentage)))]

    def calculate_compatibility(self, genotype: Genotype, threshold: float, max_diffs: int) -> Tuple[bool, float]:
        return self.representative.is_compatible(genotype, threshold, max_diffs)

    def get_elite(self, elitism: float) -> List[Genotype]:
        sort_indices = np.argsort([member.score for member in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(len(sort_indices))]

        return self.members[:math.ceil(len(self.members) * elitism)]

    def get_members_over_slot_limit(self, slots: int) -> List[Genotype]:
        outliers = []
        diff = slots - len(self.members)

        if diff < 0:
            scores_indices = np.argsort([member.score for member in self.members])[::-1]
            outliers.extend([self.members[scores_indices[i]] for i in range(slots, len(self.members))])

            # self.members = [self.members[scores_indices[i]] for i in range(slots)]
            # self.compatibility = [self.compatibility[scores_indices[i]] for i in range(slots)]

        return outliers

    def add_templates(self, config: Config, genotypes: List[Genotype]):
        species_genotypes_nodes = {}
        species_genotypes_edges = {}

        orig_members_len = len(self.members)
        skips = 0

        for i in range(len(genotypes)):
            template_genotype = self.members[(i - skips) % orig_members_len]
            genotype = genotypes[i]

            is_compatible, compatibility = self.calculate_compatibility(genotype, config.compatibility_threshold, config.compatibility_max_diffs)
            if is_compatible:
                skips += 1
            else:
                template_genotype_nodes = species_genotypes_nodes.get(template_genotype)
                if template_genotype_nodes is None:
                    template_genotype_nodes = {node.id: node.type for node in template_genotype.nodes}
                    species_genotypes_nodes[template_genotype] = template_genotype_nodes

                template_genotype_edges = species_genotypes_edges.get(template_genotype)
                if template_genotype_edges is None:
                    template_genotype_edges = {edge.innovation: (edge.input, edge.output, edge.enabled) for edge in
                                               template_genotype.edges}
                    species_genotypes_edges[template_genotype] = template_genotype_edges

                genotype_edges = {edge.innovation: (edge.input, edge.output, edge.enabled, edge.weight) for edge in
                                  genotype.edges}

                genotype.nodes = [Node(node_id, template_genotype_nodes[node_id]) for node_id in
                                  template_genotype_nodes]
                genotype.edges = []
                for edge_innovation in template_genotype_edges:
                    edge_input, edge_output, edge_enabled = template_genotype_edges[edge_innovation]
                    orig_edge = genotype_edges.get(edge_innovation)
                    genotype.edges.append(Edge(config, edge_input, edge_output, edge_enabled, edge_innovation,
                                               weight=orig_edge if orig_edge is None else orig_edge[3]))

                """
                Preserve something from template
                max_additions = math.floor(len(template_genotype_edges) * (1 + ((1 - config.compatibility_threshold) * 0.35)))
                for edge_innovation in genotype_edges:
                    edge_input, edge_output, edge_enabled, edge_weight = genotype_edges[edge_innovation]

                    if edge_input in template_genotype_nodes and edge_output in template_genotype_nodes and edge_enabled and edge_innovation not in template_genotype_edges:
                        genotype.edges.append(Edge(config, edge_input, edge_output, edge_enabled, edge_innovation, weight=edge_weight))
                        max_additions -= 1

                    if max_additions == 0:
                        break
                """
                is_compatible, compatibility = self.calculate_compatibility(genotype, config.compatibility_threshold, config.compatibility_max_diffs)

            self.add_member(genotype, compatibility)

    def get_best_member(self) -> Genotype:
        best_score = 0
        for i in range(1, len(self.members)):
            if self.members[i].score > self.members[best_score].score:
                best_score = i
        return self.members[best_score]

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def is_extinct(self) -> bool:
        return len(self.members) <= 2

    def __repr__(self):
        return "### Fitness: " + str(self.score) + ", Members: " + str(len(self.members)) + " ###"
