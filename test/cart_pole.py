from gym_client.cart_pole_client import CartPoleClient
from neat.ann.ann import Ann
import neat_tools

nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.OUTPUT), Node(5, NodeType.OUTPUT)"
edges = "0->4 (6) 0.854805581575321, 0->5 (7) 0.487770182072079, 1->4 (8) -2.4418116733189583, 1->5 (9) -1.4979210280126907, 2->4 (10) 0.09275893238164507, 2->5 (11) 4.128803832496345, 3->4 (12) -7.370591881558138"

client = CartPoleClient()

genotype = neat_tools.parse_genotype(nodes, edges)
neat_tools.visualize_genotype(genotype)
ann = Ann(genotype)

client.render(ann)
