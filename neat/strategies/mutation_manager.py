from typing import List

from neat.config import Config
from neat.encoding.genotype import Genotype
from neat.encoding.innovation_map import InnovationMap
from neat.species import Species

import numpy as np


class MutationManager:

    def __init__(self, config: Config, innovation_map: InnovationMap):
        self.config = config
        self.innovation_map = innovation_map

        self.prop_nde_pert_over_random = self.config.mutate_nde_perturbation_over_random

        self.prop_de_shigt = self.config.mutate_de_shift
        self.prop_de_pert = self.config.mutate_de_perturbate
        self.prop_de_random = self.config.mutate_de_random

        self.prop_top_add_edge = self.config.mutate_add_edge
        self.prop_top_add_node = self.config.mutate_add_node

        self.prop_node_activation_change = self.config.mutate_activation_change
        self.after_merge = False

    def restart(self, species: List[Species]):
        for s in species:
            s.reset_template()

    def mutate(self, genotype: Genotype, species: Species = None):
        self._mutate_nodes(genotype)
        self._mutate_pert_weights(genotype)
        self._mutate_de_weights(genotype)
        self._mutate_topology(genotype, species)

    def _mutate_topology(self, genotype: Genotype, species: Species = None):
        species_template = ({}, {}) if species is None else species.get_mutate_template()

        add_node = np.random.uniform() < self.prop_top_add_node
        if add_node:
            genotype.mutate_add_node(self.config, self.innovation_map, species_template)

        if np.random.uniform() < self.prop_top_add_edge and not add_node:
            genotype.mutate_add_edge(self.config, self.innovation_map, species_template)

    def _mutate_nodes(self, genotype):
        for node in genotype.nodes:
            if np.random.uniform() < self.prop_node_activation_change:
                node.mutate_activation()

    def _mutate_de_weights(self, genotype):
        for edge in genotype.edges:
            if not edge.mutable and edge.enabled:
                if np.random.uniform() < self.prop_de_shigt:
                    edge.mutate_shift_weight()

                if np.random.uniform() < self.prop_de_pert:
                    edge.mutate_perturbate_weight()

                if np.random.uniform() < self.prop_de_random:
                    edge.mutate_random_weight()

    def _mutate_pert_weights(self, genotype):
        for edge in genotype.edges:
            if edge.mutable and edge.enabled:
                if np.random.uniform() < self.prop_nde_pert_over_random:
                    edge.mutate_perturbate_weight()
                else:
                    edge.mutate_random_weight()

    def mutate_weights(self, population: List[Genotype]):
        if self.config.generation % 100 == 0 and self.config.generation > 0:
            [e.mutate_random_weight() for g in population for e in g.edges if np.random.uniform() < 0.5 and e.enabled]
            return True
        return False

    def generation_end(self):
        pass
