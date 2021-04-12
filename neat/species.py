import math
import random

import numpy as np
from typing import List, Tuple, Dict, Union, Any

from evaluators.evaluator import Evaluator
from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.encoding.innovation_map import InnovationMap
from neat.encoding.node_activation import NodeActivation
from neat.encoding.node_type import NodeType


class Species:
    """Genotype Species"""

    def __init__(self, representative: Genotype, species_id: int, max_diffs: int, elitism: float):
        self.id = species_id

        self.representative = representative

        self.members = []  # type: List[Genotype]
        self.add_member(representative, 0)

        self.age = 0
        self.fitness = 0
        self.score = 0

        self.elitism = elitism

        self.max_diffs = max_diffs

        self._nodes = {}  # type: Dict[int, Tuple[NodeType, NodeActivation]]
        self._edges = {}  # type: Dict[int, Tuple[int, int]]

    def add_member(self, genotype: Genotype, diffs: int = None) -> None:
        if genotype in self.members:
            self.members.remove(genotype)

        genotype.species_diff = self.calculate_compatibility(genotype)[1] if diffs is None else diffs
        genotype.species_id = self.id
        self.members.append(genotype)

    def reset_template(self) -> None:
        self._nodes = {}
        self._edges = {}

        for g in self.get_elite():
            for e in g.edges:
                if e.enabled and e.innovation not in self._edges:
                    self._edges[e.innovation] = (e.input, e.output)

            for n in g.nodes:
                if n.is_hidden() and n.id not in self._nodes:
                    self._nodes[n.id] = (n.type, n.activation)

    def get_mutate_template(self) -> Tuple[Dict[int, Tuple[NodeType, NodeActivation]], Dict[int, Tuple[int, int]]]:
        return self._nodes, self._edges

    def sort_members_by_score(self):
        sort_indices = np.argsort([g.score for g in self.members])[::-1]
        self.members = [self.members[sort_indices[i]] for i in range(len(sort_indices))]

    def reset(self, species: List["Species"], population: List[Genotype]):
        self.age += 1
        # Reselect old representative
        first = False  # True  # self.representative in population

        while len(self.members) > 0:
            if first:
                representative = self.representative
                first = False
            else:
                representative = self.get_best_member()
                # representative = random.sample(self.members, 1)[0]

            if representative in population:
                is_valid = True
                for s in [s for s in species if s.representative is not None]:
                    if s.calculate_compatibility(representative)[0]:
                        is_valid = False
                        break
            else:
                is_valid = False

            if is_valid:
                self.representative = representative
                self.representative.species_diff = 0
                break
            else:
                self.members.remove(representative)

        if len(self.members) > 0:
            self.members = []  # type: List[Genotype]
            self.add_member(self.representative, 0)
        else:
            self.representative = None

    def evaluate_fitness(self):
        self.fitness = 0 if len(self.members) == 0 else sum(member.fitness for member in self.members) / len(self.members)
        self.score = 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def get_score(self):
        return 0 if len(self.members) == 0 else sum(member.score for member in self.members) / len(self.members)

    def calculate_compatibility(self, genotype: Genotype) -> Tuple[bool, int]:
        return self.representative.is_compatible(genotype, self.max_diffs)

    def get_elite(self) -> List[Genotype]:
        sort_indices = np.argsort([member.score for member in self.members])[::-1]
        self.members = [self.members[i] for i in sort_indices]
        return self.members[:max(3, math.ceil(len(self.members) * self.elitism))]

    def get_species_edges(self) -> Dict[int, Tuple[int, int]]:
        edges = {}
        for genotype in self.members:
            for edge in genotype.edges:
                if edge.innovation not in edges and edge.enabled:
                    edges[edge.innovation] = (edge.input, edge.output)
        return edges

    def get_species_nodes(self) -> Dict[int, NodeType]:
        nodes = {}
        for genotype in self.members:
            for node in genotype.nodes:
                if node.id not in nodes and node.type == NodeType.HIDDEN:
                    nodes[node.id] = node.type
        return nodes

    def get_members_over_slot_limit(self, slots: int) -> List[Genotype]:
        outliers = []
        diff = slots - len(self.members)

        if diff < 0:
            scores_indices = np.argsort([member.score for member in self.members])[::-1]

            for i in range(slots, len(self.members)):
                genotype = self.members[scores_indices[i]]
                if genotype == self.representative:
                    genotype = self.members[scores_indices[slots - 1]]
                outliers.append(genotype)

            self.members = [g for g in self.members if g not in outliers]

        return outliers

    def mutate_stagnating_members(self, config: Config, innovation_map: InnovationMap) -> List[Genotype]:
        return []

        elite = self.get_elite()
        # elite = [self.get_best_member()]

        stagnating = [g for g in self.members
                      if config.generation - g.origin_generation > config.stagnation_ind and g not in elite and g != self.representative]

        for genotype in stagnating:
            random_genotype = self.roulette_select(elite, 1)[0]

            old_w = genotype.fitness / random_genotype.fitness if random_genotype.fitness != 0 else 1.0
            genotype.mutate_to_template(config, random_genotype.get_template(), min(0.75, old_w), innovation_map)
            self.add_member(genotype)

        return stagnating

    def add_templates(self, config: Config, genotypes: List[Genotype], forced_topology: bool, innovation_map: InnovationMap):
        for genotype in genotypes:
            if genotype in self.members:
                _, diffs = self.calculate_compatibility(genotype)
                genotype.origin_generation = config.generation
                genotype.ancestor = None
                self.add_member(genotype, diffs)

        best_members = self.get_elite()
        # best_members = [self.get_best_member()]

        for genotype in genotypes:
            if genotype in self.members:
                continue

            if not forced_topology:
                is_compatible, diffs = self.calculate_compatibility(genotype)
                if is_compatible:
                    genotype.origin_generation = config.generation
                    genotype.ancestor = None
                    self.add_member(genotype, diffs)
                    continue

            random_elite = self.roulette_select(best_members, 1)[0]  # type: Genotype

            # genotype.mutate_by_template(config, random_elite.get_template(), 0.03)
            # old_w = genotype.fitness / random_elite.fitness if random_elite.fitness != 0 else 1.0

            genotype.mutate_to_template(config, random_elite.get_template(), 1.0, innovation_map)

            self.add_member(genotype)

    def restart_to_best(self, config: Config, best_genotype: Genotype, population: List[Genotype], innovation_map: InnovationMap):
        best_template = best_genotype.get_template()
        best_genotype.origin_generation = config.generation
        best_genotype.ancestor = None

        for genotype in population:
            if genotype in self.members:
                continue

            old_w = genotype.fitness / best_genotype.fitness if best_genotype.fitness != 0 else 1.0
            genotype.mutate_to_template(config, best_template, old_w, innovation_map)
            self.add_member(genotype)

    def get_best_member(self) -> Genotype:
        best_score = 0
        for i in range(1, len(self.members)):
            if self.members[i].score > self.members[best_score].score:
                best_score = i
        return self.members[best_score]

    def get_best_members(self, count: int) -> List[Genotype]:
        indices = np.argsort([g.score for g in self.members])[::-1]
        return [self.members[indices[i]] for i in range(min(len(self.members), count))]

    def is_empty(self) -> bool:
        return len(self.members) == 0

    def is_extinct(self) -> bool:
        return len(self.members) <= 3

    def __repr__(self):
        return "Species id: " + str(self.id) + "; Members: " + str(len(self.members)) + "; Score: " + str(self.score) + ";"

    def test_elite(self, evaluator: Evaluator, seed: Union[None, List[Any]]) -> float:
        _, avg, _ = evaluator.test(self.get_elite(), seed)
        return sum(avg) / len(avg)

    def roulette_select(self, population: List[Genotype], count: int) -> List[Genotype]:
        selected = []
        population = list(population)

        for i in range(count):
            g = self._roulette_select_single(population)
            selected.append(g)
            population.remove(g)

        return selected

    @staticmethod
    def _roulette_select_single(population: List[Genotype]) -> Genotype:
        total = sum(g.fitness for g in population)

        if total > 0:
            props = [g.fitness / total for g in population]
        else:
            props = [1 / len(population) for _ in population]

        r = random.random()

        for i in range(len(population)):
            r -= props[i]
            if r <= 0:
                return population[i]
