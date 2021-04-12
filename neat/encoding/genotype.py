import copy
import math
import random
from typing import List, Any, Tuple, Union, Dict, Set

from neat.config import Config
from neat.encoding.innovation_map import InnovationMap
from neat.encoding.edge import Edge
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType
from neat.encoding.node_activation import NodeActivation


import numpy as np


class Genotype:

    def __init__(self, origin_generation: int, species_id: int = None, ancestor: "Genotype" = None):
        self.worse_ancestor = None
        self.ancestor = ancestor

        if ancestor is not None:
            self.worse_ancestor = ancestor.worse_ancestor
            if ancestor.ancestor is not None:
                ancestor.ancestor = None

        self.origin_generation = origin_generation

        self.nodes = []  # type: List[Node]
        self.edges = []  # type: List[Edge]

        self.fitness = 0
        self.score = 0
        self.evaluated_seed = []  # type: Union[List[Any], None]
        self.scores = []  # type: List[float]

        self.species_diff = 0

        if species_id is None and ancestor is not None:
            species_id = ancestor.species_id

        self.species_id = species_id
        self.curr_orig_species_id = species_id

    def get_node_by_id(self, input_node_id: int) -> Node:
        for n in self.nodes:
            if n.id == input_node_id:
                return n

    def get_edge_by_innovation(self, innovation: int) -> Edge:
        for e in self.edges:
            if e.innovation == innovation:
                return e

    @staticmethod
    def init_population(config: Config, innovation_map: InnovationMap) -> List["Genotype"]:
        initial_genotype = Genotype.__init_genotype(config, innovation_map)
        population = [initial_genotype.__init_copy() for _ in range(config.population_size)]  # type: List[Genotype]

        if not config.start_min:
            for g in population:
                g.mutate_add_edge(config, innovation_map, ({}, {}))
                g.mutate_add_edge(config, innovation_map, ({}, {}))

        return population

    @staticmethod
    def __init_genotype(config: Config, innovation_map: InnovationMap) -> "Genotype":
        genotype = Genotype(config.generation)

        for i in range(config.input_nodes + config.bias_nodes):
            innovation = innovation_map.next_innovation()
            genotype.nodes.append(Node(innovation, NodeType.INPUT))

        for i in range(config.output_nodes):
            innovation = innovation_map.next_innovation()
            genotype.nodes.append(Node(innovation, NodeType.OUTPUT, None, i))

        return genotype

    @staticmethod
    def init_population_from_best_topology(config: Config, genotype: "Genotype") -> List["Genotype"]:
        genotype.origin_generation = config.generation

        genotype.fitness = 0
        genotype.score = 0
        genotype.evaluated_seed = []
        genotype.scores = []

        genotype.species_diff = 0
        genotype.min_score = 0

        genotype.species_id = None
        genotype.curr_orig_species_id = None

        return [genotype.__init_copy() for _ in range(config.population_size)]

    def copy(self) -> "Genotype":
        ancestor = self.ancestor
        self.ancestor = None
        cpy = copy.deepcopy(self)
        self.ancestor = ancestor
        return cpy

    def mutation_copy(self, generation) -> "Genotype":
        ancestor, self.ancestor = self.ancestor, None

        genotype = copy.deepcopy(self)
        genotype.reset(generation, None, self)

        self.ancestor = ancestor
        return genotype

    def reset(self, generation: int, species_id: int = None, ancestor: "Genotype" = None):
        self.ancestor = ancestor
        if ancestor is not None:
            if ancestor.ancestor is not None:
                ancestor.ancestor = None

        self.origin_generation = generation
        self.fitness = 0
        self.score = 0
        self.species_diff = 0

        self.evaluated_seed = []  # type: Union[List[Any], None]
        self.scores = []  # type: List[float]

        if species_id is None and ancestor is not None:
            species_id = ancestor.species_id

        self.species_id = species_id
        self.curr_orig_species_id = species_id

    def __init_copy(self) -> "Genotype":
        cp = copy.deepcopy(self)
        for e in cp.edges:
            e.mutate_random_weight()
        return cp

    def mutate_add_node(self, config: Config,
                        innovation_map: InnovationMap,
                        mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
                        forced_random: bool = False):

        self._mutate_add_node_random_insert(config, innovation_map, mutate_template, forced_random)
        # self._mutate_add_node_disjoint_edge(config, innovation_map, mutate_template, forced_random)

    # def _mutate_add_node_random_insert_with_last_layer(self, config: Config,
    #                                                    innovation_map: InnovationMap,
    #                                                    mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
    #                                                    forced_random: bool = False):
    #
    #     curr_nodes, curr_edges = self.get_template()
    #     node_layers, layers_list = self.get_node_layers()
    #
    #     species_mutations = mutate_template[0]
    #     edge_mutations = mutate_template[1]
    #     possible_mutations = []
    #
    #     random_mutation = forced_random or self._get_mutate_topology_random_prop(config)
    #     node_activation = None
    #
    #     neurons_inputs_sizes = {n.id: 0 for n in self.nodes if n.is_output()}  # type: Dict[int, int]
    #     for edge in self.edges:
    #         if edge.output in neurons_inputs_sizes:
    #             neurons_inputs_sizes[edge.output] += 1
    #     possible_extends = {n_id for n_id in neurons_inputs_sizes if neurons_inputs_sizes[n_id] >= 2}
    #
    #     if not random_mutation:
    #         for node_id in species_mutations:
    #             # Random hidden_node
    #             input_id, output_id = innovation_map.get_node_direction(node_id)
    #             if node_id not in curr_nodes and input_id in curr_nodes and output_id in curr_nodes and node_layers[output_id] >= node_layers[input_id]:
    #                 possible_mutations.append((0, input_id, output_id, species_mutations[node_id][1]))
    #
    #             # Random output node
    #             if node_id in possible_extends:
    #                 innovation = innovation_map.next_extend_innovation(node_id, self.nodes)
    #                 if innovation in species_mutations:
    #                     for edge_innovation in edge_mutations:
    #                         i, o = edge_mutations[edge_innovation]
    #                         if i == node_id:
    #                             possible_mutations.append((1, i, o))
    #                             break
    #
    #     hidden_mutation = False
    #     output_mutation = False
    #
    #     if random_mutation or len(possible_mutations) == 0:
    #         for i in range(1000):
    #             samples = random.sample(self.nodes, 2)  # type: List[Node]
    #             in_node, out_node = samples[0], samples[1]
    #             input_node, output_node = samples[0].id, samples[1].id
    #             input_layer, output_layer = node_layers[input_node], node_layers[output_node]
    #
    #             if in_node.is_output() and out_node.is_output() and input_node in possible_extends:
    #                 output_mutation = True
    #                 break
    #             elif output_layer > input_layer or (output_layer == input_layer and input_layer != 0 and input_layer != len(layers_list) - 1):
    #                 hidden_mutation = True
    #                 break
    #     else:
    #         mutation = random.sample(possible_mutations, 1)[0]
    #
    #         if mutation[0] == 0:
    #             hidden_mutation = True
    #             _, input_node, output_node, node_activation = mutation
    #         elif mutation[1] == 1:
    #             output_mutation = True
    #             _, input_node, output_node = mutation
    #             in_node = self.get_node_by_id(input_node)
    #
    #     if hidden_mutation:
    #         # Create new node
    #         node_innovation = innovation_map.get_node_innovation(input_node, output_node, self.nodes)
    #         new_node = Node(node_innovation, NodeType.HIDDEN, node_activation)
    #         self.nodes.append(new_node)
    #
    #         # Create new connections
    #         self.edges.append(Edge(self, config, input_node, new_node.id, True, innovation_map.get_edge_innovation(input_node, new_node.id)))
    #         self.edges.append(Edge(self, config, new_node.id, output_node, True, innovation_map.get_edge_innovation(new_node.id, output_node)))
    #
    #     elif output_mutation:
    #         # Create new node
    #         innovation = innovation_map.next_extend_innovation(input_node, self.nodes)
    #         new_node = Node(innovation, in_node.type, in_node.activation, in_node.output_id)
    #         self.nodes.append(new_node)
    #
    #         # Modify old node
    #         in_node.type = NodeType.HIDDEN
    #         in_node.activation = NodeActivation.LIN
    #         in_node.output_id = None
    #
    #         # Create new edge to connect nodes
    #         self.edges.append(Edge(self, config, input_node, new_node.id, True,
    #                                innovation_map.get_edge_innovation(input_node, new_node.id), weight=np.random.uniform(0.97, 1.03)))
    #
    #         self.edges.append(Edge(self, config, new_node.id, output_node, True,
    #                                innovation_map.get_edge_innovation(new_node.id, output_node), weight=np.random.uniform(-0.03, 0.03)))
    #
    # def _mutate_add_node_dual(self, config: Config,
    #                           innovation_map: InnovationMap,
    #                           mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
    #                           forced_random: bool = False):
    #     random_mutation = forced_random or self._get_mutate_topology_random_prop(config)
    #     layers = self.get_node_layers()
    #
    #     if random_mutation:
    #         self._mutate_add_node_dual_random(layers, config, innovation_map)
    #     else:
    #         if not self._mutate_add_node_dual_template(layers, config, innovation_map, mutate_template):
    #             self._mutate_add_node_dual_random(layers, config, innovation_map)
    #
    # def _mutate_add_node_dual_random(self, layers: Tuple[Dict[int, int], List[List[int]]], config: Config,
    #                                  innovation_map: InnovationMap):
    #     input_node = random.sample(self.nodes, 1)[0]  # type: Node
    #
    #     if input_node.is_output():
    #         # Create new node
    #         innovation = innovation_map.next_extend_innovation(input_node.id, self.nodes)
    #         new_node = Node(innovation, input_node.type, input_node.activation, input_node.output_id)
    #         self.nodes.append(new_node)
    #
    #         # Modify old node
    #         input_node.type = NodeType.HIDDEN
    #         input_node.activation = NodeActivation.LIN
    #         input_node.output_id = None
    #
    #         # Create new edge to connect nodes
    #         weight = np.random.uniform(0.9, 1.1)
    #         self.edges.append(Edge(self, config, input_node.id, new_node.id, True,
    #                                innovation_map.get_edge_innovation(input_node.id, new_node.id), weight=weight))
    #     else:
    #         layer_from = 1 if input_node.is_input() or input_node.is_bias() else layers[0][input_node.id]
    #         sample_outputs = [neuron_id for i in range(layer_from, len(layers[1])) for neuron_id in layers[1][i]]
    #         output_node = self.get_node_by_id(random.sample(sample_outputs, 1)[0])  # type: Node
    #
    #         # Create new node
    #         node_innovation = innovation_map.get_node_innovation(input_node.id, output_node.id, self.nodes)
    #         new_node = Node(node_innovation, NodeType.HIDDEN)
    #         self.nodes.append(new_node)
    #
    #         # Create new connections
    #         self.edges.append(Edge(self, config, input_node.id, new_node.id, True, innovation_map.get_edge_innovation(input_node.id, new_node.id)))
    #         self.edges.append(Edge(self, config, new_node.id, output_node.id, True, innovation_map.get_edge_innovation(new_node.id, output_node.id)))
    #
    # def _mutate_add_node_dual_template(self, layers: Tuple[Dict[int, int], List[List[int]]], config: Config,
    #                                    innovation_map: InnovationMap,
    #                                    mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]]):
    #     curr_nodes, curr_edges = self.get_template()
    #     node_layers, layers_list = self.get_node_layers()
    #
    #     species_mutations = mutate_template[0]
    #     possible_mutations = []
    #
    #     neurons_inputs_sizes = {n.id: 0 for n in self.nodes if n.is_output()}  # type: Dict[int, int]
    #     for edge in self.edges:
    #         if edge.output in neurons_inputs_sizes:
    #             neurons_inputs_sizes[edge.output] += 1
    #
    #     possible_extends = list(n_id for n_id in neurons_inputs_sizes if neurons_inputs_sizes[n_id] >= 2)
    #
    #     for node_id in species_mutations:
    #         # Random hidden node
    #         input_id, output_id = innovation_map.get_node_direction(node_id)
    #         if node_id not in curr_nodes and input_id in curr_nodes and output_id in curr_nodes and node_layers[output_id] >= node_layers[input_id]:
    #             possible_mutations.append((0, input_id, output_id, species_mutations[node_id][1]))
    #
    #         # Random output node
    #         if node_id in possible_extends:
    #             innovation = innovation_map.next_extend_innovation(node_id, self.nodes)
    #             if innovation in species_mutations:
    #                 possible_mutations.append((1, node_id))
    #
    #     if len(possible_mutations) == 0:
    #         return False
    #
    #     mutation = random.sample(possible_mutations, 1)[0]
    #     if mutation[0] == 0:
    #         _, input_node_id, output_node_id, node_activation = mutation
    #
    #         # Create new node
    #         node_innovation = innovation_map.get_node_innovation(input_node_id, output_node_id, self.nodes)
    #         new_node = Node(node_innovation, NodeType.HIDDEN, node_activation)
    #         self.nodes.append(new_node)
    #
    #         # Create new connections
    #         self.edges.append(Edge(self, config, input_node_id, new_node.id, True, innovation_map.get_edge_innovation(input_node_id, new_node.id)))
    #         self.edges.append(Edge(self, config, new_node.id, output_node_id, True, innovation_map.get_edge_innovation(new_node.id, output_node_id)))
    #     else:
    #         extend_node = self.get_node_by_id(mutation[1])
    #
    #         # Create new node
    #         innovation = innovation_map.next_extend_innovation(extend_node.id, self.nodes)
    #         new_node = Node(innovation, extend_node.type, extend_node.activation, extend_node.output_id)
    #         self.nodes.append(new_node)
    #
    #         # Modify old node
    #         extend_node.type = NodeType.HIDDEN
    #         extend_node.activation = NodeActivation.LIN
    #         extend_node.output_id = None
    #
    #         # Create new edge to connect nodes
    #         weight = np.random.uniform(0.9, 1.1)
    #         self.edges.append(
    #             Edge(self, config, extend_node.id, new_node.id, True,
    #                  innovation_map.get_edge_innovation(extend_node.id, new_node.id), weight=weight))
    #
    #     return True
    #
    # def _mutate_add_node_extend(self, config: Config,
    #                             innovation_map: InnovationMap,
    #                             mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
    #                             forced_random: bool = False) -> bool:
    #     if len(self.edges) == 0:
    #         return False
    #
    #     neurons_inputs_sizes = {n.id: 0 for n in self.nodes if n.is_output()}  # type: Dict[int, int]
    #     for edge in self.edges:
    #         if edge.output in neurons_inputs_sizes:
    #             neurons_inputs_sizes[edge.output] += 1
    #
    #     possible_extends = list(n_id for n_id in neurons_inputs_sizes if neurons_inputs_sizes[n_id] >= 2)
    #     if len(possible_extends) == 0:
    #         return False
    #
    #     curr_nodes, curr_edges = self.get_template()
    #     species_mutations = mutate_template[0]
    #
    #     possible_mutations = []  # type: List[int]
    #     random_mutation = forced_random or self._get_mutate_topology_random_prop(config)
    #
    #     if not random_mutation:
    #         for node_id in species_mutations:
    #             if node_id in possible_extends:
    #                 innovation = innovation_map.next_extend_innovation(node_id, self.nodes)
    #                 if innovation in species_mutations:
    #                     possible_mutations.append(node_id)
    #
    #     if random_mutation or len(possible_mutations) == 0:
    #         extend_node_id = random.sample(possible_extends, 1)[0]
    #         extend_node = self.get_node_by_id(extend_node_id)
    #     else:
    #         extend_node_id = random.sample(possible_mutations, 1)[0]
    #         extend_node = self.get_node_by_id(extend_node_id)
    #
    #     # Create new node
    #     innovation = innovation_map.next_extend_innovation(extend_node_id, self.nodes)
    #     new_node = Node(innovation, extend_node.type, extend_node.activation, extend_node.output_id)
    #     self.nodes.append(new_node)
    #
    #     # Modify old node
    #     extend_node.type = NodeType.HIDDEN
    #     extend_node.activation = NodeActivation.LIN
    #     extend_node.output_id = None
    #
    #     # Create new edge to connect nodes
    #     weight = np.random.uniform(0.9, 1.1)
    #     self.edges.append(
    #         Edge(self, config, extend_node.id, new_node.id, True,
    #              innovation_map.get_edge_innovation(extend_node.id, new_node.id), weight=weight))
    #
    #     return True
    #

    def _mutate_add_node_disjoint_edge(self, config: Config,
                                       innovation_map: InnovationMap,
                                       mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
                                       forced_random: bool = False):
        if len(self.edges) == 0:
            return

        curr_nodes, curr_edges = self.get_template()
        node_layers, _ = self.get_node_layers()

        species_mutations = mutate_template[0]
        possible_mutations = []

        random_mutation = True  # self._get_mutate_topology_random_prop(config) or forced_random
        node_activation = None

        if not random_mutation:
            for node_id in species_mutations:
                input_id, output_id = innovation_map.get_node_direction(node_id)
                if node_id not in curr_nodes and input_id in curr_nodes and output_id in curr_nodes and node_layers[output_id] >= node_layers[input_id]:
                    disjoint_edge_innovation = innovation_map.get_edge_innovation(input_id, output_id)
                    if disjoint_edge_innovation in curr_edges:
                        possible_mutations.append((disjoint_edge_innovation, species_mutations[node_id][1]))

        if random_mutation or len(possible_mutations) == 0:
            disjoint_edge = random.sample([e for e in self.edges if e.enabled], 1)[0]
        else:
            disjoint_edge_innovation, node_activation = random.sample(possible_mutations, 1)[0]
            disjoint_edge = self.get_edge_by_innovation(disjoint_edge_innovation)

        # Create new node
        node_innovation = innovation_map.get_node_innovation(disjoint_edge.input, disjoint_edge.output, self.nodes)
        new_node = Node(node_innovation, NodeType.HIDDEN, node_activation)
        self.nodes.append(new_node)

        # Create new connections
        self.edges.append(Edge(self, config, disjoint_edge.input, new_node.id, True, innovation_map.get_edge_innovation(disjoint_edge.input, new_node.id)))
        self.edges.append(Edge(self, config, new_node.id, disjoint_edge.output, True, innovation_map.get_edge_innovation(new_node.id, disjoint_edge.output)))

        disjoint_edge.enabled = False

        # Remove old connection
        # self.edges.remove(disjoint_edge)

    def _mutate_add_node_random_insert(self, config: Config,
                                       innovation_map: InnovationMap,
                                       mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
                                       forced_random: bool = False):

        curr_nodes, curr_edges = self.get_template()
        node_layers, layers_list = self.get_node_layers()

        species_mutations = mutate_template[0]
        possible_mutations = []

        random_mutation = True  # forced_random or self._get_mutate_topology_random_prop(config)
        node_activation = None

        if not random_mutation:
            for node_id in species_mutations:
                input_id, output_id = innovation_map.get_node_direction(node_id)
                if node_id not in curr_nodes and input_id in curr_nodes and output_id in curr_nodes and node_layers[output_id] >= node_layers[input_id]:
                    possible_mutations.append((input_id, output_id, species_mutations[node_id][1]))

        hidden_mutation = False
        if random_mutation or len(possible_mutations) == 0:
            nodes_input_count = self.get_nodes_input_count()  # type: Dict[int, int]
            for i in range(1000):
                # samples = random.sample(self.nodes, 2)  # type: List[Node]
                samples = self.sample_nodes_by_inputs(nodes_input_count)  # type: List[Node]

                input_node, output_node = samples[0].id, samples[1].id
                input_layer, output_layer = node_layers[input_node], node_layers[output_node]

                if output_layer > input_layer or (output_layer == input_layer and input_layer != 0 and input_layer != len(layers_list) - 1):
                    hidden_mutation = True
                    break
        else:
            input_node, output_node, node_activation = random.sample(possible_mutations, 1)[0]
            hidden_mutation = True

        if hidden_mutation:
            # Create new node
            node_innovation = innovation_map.get_node_innovation(input_node, output_node, self.nodes)
            new_node = Node(node_innovation, NodeType.HIDDEN, node_activation)
            self.nodes.append(new_node)

            # Create new connections
            self.edges.append(Edge(self, config, input_node, new_node.id, True, innovation_map.get_edge_innovation(input_node, new_node.id)))
            self.edges.append(Edge(self, config, new_node.id, output_node, True, innovation_map.get_edge_innovation(new_node.id, output_node)))

    def mutate_add_edge(self, config: Config,
                        innovation_map: InnovationMap,
                        mutate_template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int]]],
                        forced_random: bool = True):

        curr_nodes, curr_edges = self.get_template()
        node_layers, layers_list = self.get_node_layers()

        species_nodes = mutate_template[0]
        species_mutations = mutate_template[1]
        possible_mutations = []

        neurons_inputs_sizes = {n.id: 0 for n in self.nodes if n.is_output()}  # type: Dict[int, int]
        for edge in self.edges:
            if edge.output in neurons_inputs_sizes:
                neurons_inputs_sizes[edge.output] += 1
        possible_extends = {n_id for n_id in neurons_inputs_sizes if neurons_inputs_sizes[n_id] >= 2}

        random_mutation = True  # forced_random or self._get_mutate_topology_random_prop(config)

        if not random_mutation:
            output_neurons = {n.id for n in self.nodes if n.is_output()}

            for innovation in species_mutations:
                input_id, output_id = species_mutations[innovation]
                if innovation not in curr_edges and input_id in output_neurons:
                    if innovation in possible_extends:
                        next_innovation = innovation_map.next_extend_innovation(input_id, self.nodes)
                        if output_id != next_innovation:
                            possible_mutations.append((1, input_id, output_id))
                elif innovation not in curr_edges and input_id in curr_nodes and output_id in curr_nodes and node_layers[output_id] >= node_layers[input_id]:
                    possible_mutations.append((0, input_id, output_id, innovation))

        hidden_mutation = False
        output_mutation = False

        if random_mutation or len(possible_mutations) == 0:
            nodes_input_count = self.get_nodes_input_count()  # type: Dict[int, int]
            for i in range(1000):
                # random.sample(self.nodes, 2)  # type: List[Node]
                samples = self.sample_nodes_by_inputs(nodes_input_count)  # type: List[Node]

                input_node, output_node = samples[0].id, samples[1].id
                input_layer, output_layer = node_layers[input_node], node_layers[output_node]

                if samples[0].is_output() and samples[1].is_output() and input_node in possible_extends:
                    output_mutation = True
                    break

                if output_layer > input_layer or (output_layer == input_layer and input_layer != 0 and input_layer != len(layers_list) - 1):
                    innovation = innovation_map.get_edge_innovation(input_node, output_node)
                    if innovation in curr_edges:
                        e = self.get_edge_by_innovation(innovation)
                        if not e.enabled:
                            e.enabled = True
                            e.mutate_random_weight()
                            return False
                    else:
                        hidden_mutation = True
                        break
        else:
            mutation = random.sample(possible_mutations, 1)[0]

            if mutation[0] == 0:
                hidden_mutation = True
                _, input_node, output_node, _ = mutation
                innovation = innovation_map.get_edge_innovation(input_node, output_node)
            elif mutation[0] == 1:
                output_mutation = True
                _, input_node, output_node = mutation

        if hidden_mutation:
            self.edges.append(Edge(self, config, input_node, output_node, True, innovation))

        if output_mutation:
            in_node = self.get_node_by_id(input_node)
            # Create new node
            innovation = innovation_map.next_extend_innovation(input_node, self.nodes)
            new_node = Node(innovation, in_node.type, in_node.activation, in_node.output_id)
            self.nodes.append(new_node)

            # Modify old node
            in_node.type = NodeType.HIDDEN
            in_node.activation = NodeActivation.LIN
            in_node.output_id = None

            # Create new edge to connect nodes
            self.edges.append(Edge(self, config, input_node, new_node.id, True,
                                   innovation_map.get_edge_innovation(input_node, new_node.id), weight=np.random.uniform(0.97, 1.03)))

            self.edges.append(Edge(self, config, input_node, output_node, True,
                                   innovation_map.get_edge_innovation(new_node.id, output_node), weight=np.random.uniform(-0.03, 0.03)))

        return output_mutation

    def get_nodes_input_count(self) -> Dict[int, int]:
        nodes = {n.id: 1 for n in self.nodes}
        for e in self.edges:
            if e.enabled and nodes.get(e.output) is not None:
                nodes[e.output] += 1
            if e.enabled and nodes.get(e.input) is not None:
                nodes[e.input] += 1
        return nodes

    def sample_nodes_by_inputs(self, nodes_input_count: Dict[int, int]) -> List[Node]:
        nodes_prop = [(node_id, 1/nodes_input_count[node_id]) for node_id in nodes_input_count]
        prop_sum = sum(n[1] for n in nodes_prop)

        if len(nodes_prop) <= 1:
            raise Exception("Not enough nodes to sample!")
        else:
            first_id = None
            rand = np.random.uniform(0, prop_sum)
            for node_id, node_prop in nodes_prop:
                if rand < node_prop:
                    first_id = node_id
                    break
                else:
                    rand -= node_prop

            second_id = first_id
            while first_id == second_id:
                rand = np.random.uniform(0, prop_sum)
                for node_id, node_prop in nodes_prop:
                    if rand < node_prop:
                        second_id = node_id
                        break
                    else:
                        rand -= node_prop

            return [self.get_node_by_id(first_id), self.get_node_by_id(second_id)]

    def _is_node_connection_possible(self, input_node: Node, output_node: Node):
        if input_node.type == NodeType.OUTPUT or (output_node.type == NodeType.INPUT or output_node.type == NodeType.BIAS):
            return False

        # NOT WORKING, check also if recurrent

        nodes = {n.id for n in self.nodes}
        if input_node.id not in nodes or output_node.id not in nodes:
            return False

        return True

    def _is_edge_connection_possible(self, input_node: Node, output_node: Node):
        if not self._is_node_connection_possible(input_node, output_node):
            return False

        for e in self.edges:
            if (e.input == input_node.id and e.output == output_node.id) or \
                    (e.input == output_node.id and e.output == input_node.id):
                return False

        return True

    def _is_new_edge_recurrent(self, input_node: int, output_node: int) -> bool:
        # Search for path from output_node to input_node
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

    @staticmethod
    def crossover_triple(generation: int, f: float, cr: float, ancestor: "Genotype", best: "Genotype", mom: "Genotype", dad: "Genotype") -> "Genotype":
        genotype = Genotype(generation, ancestor.species_id, ancestor)

        best_edges = {edge.innovation: edge.weight for edge in best.edges if edge.enabled}
        mom_edges = {edge.innovation: edge.weight for edge in mom.edges if edge.enabled}
        dad_edges = {edge.innovation: edge.weight for edge in dad.edges if edge.enabled}

        genotype.nodes = [n.copy() for n in ancestor.nodes]

        for edge in ancestor.edges:
            c = edge.weight
            b = best_edges.get(edge.innovation)
            m = mom_edges.get(edge.innovation)
            d = dad_edges.get(edge.innovation)

            if b is None or m is None or d is None or not edge.enabled or np.random.uniform() > cr:
                genotype.edges.append(Edge(genotype, edge.config, edge.input, edge.output, edge.enabled, edge.innovation, edge.enabled, c))
            else:
                genotype.edges.append(Edge(genotype, edge.config, edge.input, edge.output, edge.enabled, edge.innovation, False, c + (b - c) * f + (m - d) * f))

        return genotype

    @staticmethod
    def crossover(generation: int, f: float, cr: float, ancestor: "Genotype", best: "Genotype", mom: "Genotype", dad: "Genotype") -> "Genotype":
        genotype = Genotype(generation, ancestor.species_id, ancestor)

        ancestor_edges = {edge.innovation: edge.weight for edge in ancestor.edges if edge.enabled}
        mom_edges = {edge.innovation: edge.weight for edge in mom.edges if edge.enabled}
        dad_edges = {edge.innovation: edge.weight for edge in dad.edges if edge.enabled}

        genotype.nodes = [n.copy() for n in best.nodes]

        for edge in best.edges:
            b = edge.weight
            c = ancestor_edges.get(edge.innovation)
            m = mom_edges.get(edge.innovation)
            d = dad_edges.get(edge.innovation)

            if m is None or d is None or not edge.enabled:
                weight = c if np.random.uniform() > cr and c is not None else edge.weight
                genotype.edges.append(Edge(genotype, edge.config, edge.input, edge.output, edge.enabled, edge.innovation, edge.enabled, weight))
            else:
                genotype.edges.append(Edge(genotype, edge.config, edge.input, edge.output, edge.enabled, edge.innovation, False, b + (m - d) * f))

        return genotype

    def get_template(self) -> Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int, bool, float]]]:
        nodes = {n.id: (n.type, n.activation, n.output_id) for n in self.nodes}
        edges = {e.innovation: (e.input, e.output, e.enabled, e.weight) for e in self.edges}

        return nodes, edges

    def mutate_to_template(self, config: Config, template: Tuple[Dict[int, Tuple[NodeType, NodeActivation, int]], Dict[int, Tuple[int, int, bool, float]]], mutation_prop: float, innovation_map: InnovationMap):
        nodes, edges = template
        self.nodes = [Node(node_id, nodes[node_id][0], nodes[node_id][1], nodes[node_id][2]) for node_id in nodes]

        original_edges = {edge.innovation: (edge.input, edge.output, edge.enabled, edge.weight) for edge in self.edges if edge.innovation in edges}
        self.edges = []

        orig_mutations = []
        if len(edges) >= 1:
            mutations_max = math.ceil(len(edges) * mutation_prop)
            orig_mutations = random.sample(list(original_edges), min(len(original_edges), mutations_max))
        orig_mutations = set(orig_mutations)

        for edge_innovation in edges:
            edge_input, edge_output, edge_enabled, edge_weight = edges[edge_innovation]

            is_original = edge_innovation in orig_mutations
            weight = original_edges[edge_innovation][3] if is_original else edge_weight

            new_edge = Edge(self, config, edge_input, edge_output, edge_enabled, edge_innovation, False, weight)
            self.edges.append(new_edge)

            if not is_original and new_edge.enabled:
                new_edge.mutate_shift_weight() if np.random.uniform() < 0.5 else new_edge.mutate_perturbate_weight()

        self.origin_generation = config.generation
        self.ancestor = None
        self.worse_ancestor = None

    def is_better(self) -> bool:
        is_better_than_ancestor = self._is_better_than_ancestor()

        # if self.worse_ancestor is not None and self.score >= self.worse_ancestor.score:
        #     self.worse_ancestor = None

        return is_better_than_ancestor

    def _is_better_than_ancestor(self) -> bool:
        if self.ancestor is None:
            return True

        elif self.score >= self.ancestor.score:
            return True

        elif self.get_topology_size() > self.ancestor.get_topology_size():  # and self.ancestor.worse_ancestor is None:
            worse_ratio = self.fitness / (self.fitness + self.ancestor.fitness)
            append_worse = worse_ratio >= 0.4 and np.random.uniform() < worse_ratio

            # if append_worse:
            #     self.worse_ancestor = self.ancestor

            return append_worse

        else:
            return False

    def get_topology_size(self) -> int:
        return sum(1 for e in self.edges if e.enabled)

    def is_compatible(self, other: "Genotype", max_diffs: int) -> Tuple[bool, int]:
        return self.__is_compatible(self, other, max_diffs)

    def calculate_compatibility(self, other: "Genotype", max_diffs: int) -> Tuple[float, int]:
        g0_innovations = {e.innovation for e in self.edges}
        g1_innovations = {e.innovation for e in other.edges}

        bigger_genotype_size = max(len(g0_innovations), len(g1_innovations))
        if len(g0_innovations) == 0 or len(g1_innovations) == 0:
            return 1 / (1 + bigger_genotype_size), bigger_genotype_size

        all_innovations = {e.innovation for e in self.edges}
        for e in other.edges:
            if e not in all_innovations:
                all_innovations.add(e.innovation)

        m = 0  # Matching genes
        d = 0  # Disjoint genes

        for i in all_innovations:
            if i in g0_innovations and i in g1_innovations:
                m += 1
            else:
                d += 1

        return m / bigger_genotype_size, d

    @staticmethod
    def __is_compatible(g0: "Genotype", g1: "Genotype", max_diffs: int) -> Tuple[bool, int]:
        g0_innovations = {e.innovation for e in g0.edges}
        g1_innovations = {e.innovation for e in g1.edges}

        if len(g0_innovations) == 0 or len(g1_innovations) == 0:
            diff = max(len(g0_innovations), len(g1_innovations))
            return diff <= max_diffs, diff

        all_innovations = {e.innovation for e in g0.edges}
        for e in g1.edges:
            if e not in all_innovations:
                all_innovations.add(e.innovation)

        m = 0  # Matching genes
        d = 0  # Disjoint genes

        for i in all_innovations:
            if i in g0_innovations and i in g1_innovations:
                m += 1
            else:
                d += 1

        return d <= max_diffs, d

    def get_max_innovation(self) -> int:
        return max(e.innovation for e in self.edges)

    def _get_mutate_topology_random_prop(self, config: Config) -> bool:
        return np.random.uniform() > (self.species_diff / config.compatibility_max_diffs)

    def get_node_layers(self) -> Tuple[Dict[int, int], List[List[int]]]:
        node_layers = {}  # node_id: node_layer type: Dict[int, int]

        end_nodes = {n.id for n in self.nodes if n.is_output()}  # to terminate type: Set[int]
        node_outputs = {n.id: set() for n in self.nodes if not n.is_output()}  # type: Dict[int, Set]

        for e in self.edges:
            if e.enabled and e.output not in end_nodes:
                node_outputs[e.input].add(e.output)

        curr_layer = 1
        first_layer = [n.id for n in self.nodes if n.is_input() or n.is_bias()]

        while len(first_layer) >= 1:
            new_first_layer = []

            for node_id in first_layer:
                for output_id in node_outputs.get(node_id):
                    new_first_layer.append(output_id)
                    node_layers[output_id] = curr_layer

            curr_layer += 1
            first_layer = new_first_layer

        last_layer_ids = {}
        for n in self.nodes:
            if n.is_input() or n.is_bias():
                node_layers[n.id] = 0
            elif n.is_output():
                node_layers[n.id] = curr_layer - 1
                last_layer_ids[n.output_id] = n.id

        ids_positions = list(last_layer_ids)
        sort_id = np.argsort(ids_positions)

        final_layers = [[] for _ in range(curr_layer)]
        for n_id in node_layers:
            final_layers[node_layers[n_id]].append(n_id)

        final_layers[-1] = [last_layer_ids[ids_positions[index]] for index in sort_id]
        return node_layers, final_layers

    def __str__(self):
        return "Score = " + str(round(self.score, 2)).ljust(6) + "; Fitness = " + str(round(self.fitness, 2)).ljust(6) + "; N = " + str(sum(1 for n in self.nodes if n.is_hidden() or n.is_output())) + "; E = " + str(sum([1 for edge in self.edges if edge.enabled]))

    def __repr__(self):
        return str(self)
