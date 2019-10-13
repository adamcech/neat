from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.genotype import Genotype
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType

import networkx as nx
from matplotlib import pyplot as plt


def parse_genotype(nodes: str, edges: str) -> Genotype:
    genotype = Genotype()

    nodes = nodes.replace("Node", "").split(", ")

    for i in range(0, len(nodes), 2):
        node_id = int(nodes[i].replace("(", ""))
        node_type = nodes[i + 1].replace(")", "")

        if node_type == "Type.INPUT":
            node_type = NodeType.INPUT
        elif node_type == "Type.OUTPUT":
            node_type = NodeType.OUTPUT
        else:
            node_type = NodeType.HIDDEN

        genotype.nodes.append(Node(node_id, node_type))

    edges = edges.split(", ")
    for edge in edges:
        params = edge.split(" ")

        if params[0] == "":
            continue

        edge_input = int(params[0].split("->")[0])
        edge_output = int(params[0].split("->")[1])
        edge_innovation = int(params[1].replace("(", "").replace(")", ""))
        edge_enabled = True
        edge_weight = float(params[2])
        genotype.edges.append(Edge(edge_input, edge_output, edge_enabled, edge_innovation, weight=edge_weight))

    return genotype


def parse_ann(nodes: str, edges: str) -> Ann:
    return Ann(parse_genotype(edges, nodes))


def visualize_genotype(genotype: Genotype):
    G = nx.MultiDiGraph()

    for node in genotype.nodes:
        if node.type == NodeType.INPUT:
            G.add_node(str(node.id))
        if node.type == NodeType.HIDDEN:
            G.add_node(str(node.id))
        if node.type == NodeType.OUTPUT:
            G.add_node(str(node.id))

    for edge in [edge for edge in genotype.edges if edge.enabled]:
        G.add_edge(str(edge.input), str(edge.output), weight=edge.weight)

    node_color = []

    for node in genotype.nodes:
        if node.type == NodeType.INPUT:
            node_color.append("green")
        if node.type == NodeType.HIDDEN:
            node_color.append("red")
        if node.type == NodeType.OUTPUT:
            node_color.append("blue")

    pos = nx.fruchterman_reingold_layout(G)
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

    nx.draw(G, pos=pos, node_size=50, with_labels=False, node_color=node_color, line_color='grey', linewidths=0,
            width=0.1)
    plt.show()
