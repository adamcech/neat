from gym_client.lunar_lander_client import LunarLanderClient
from neat.ann.ann import Ann
import neat_tools

from neat.neat import Neat


def main():
    nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.INPUT), Node(5, NodeType.INPUT), Node(6, NodeType.INPUT), Node(7, NodeType.INPUT), Node(8, NodeType.OUTPUT), Node(9, NodeType.OUTPUT), Node(10, NodeType.OUTPUT), Node(11, NodeType.OUTPUT), Node(143, NodeType.HIDDEN), Node(80, NodeType.HIDDEN), Node(59, NodeType.HIDDEN), Node(89, NodeType.HIDDEN), Node(120, NodeType.HIDDEN)"
    edges = "0->8 (12) 1.5785251784856427, 0->9 (13) 2.2245588640626384, , 0->11 (15) -1.701268112981892, 1->8 (16) -8, 1->9 (17) 2.5499171038995976, , 1->11 (19) 2.0209066125509922, 2->8 (20) -1.5541538338101013, 2->9 (21) 7.974702538493737, 2->10 (22) 0.4508108938847992, 2->11 (23) -1.5190508403936769, 3->8 (24) -0.09325306992249338, 3->9 (25) 4.811555794333891, 3->10 (26) -8, , 4->8 (28) 1.5979081139529108, 4->9 (29) -5.562176655068532, 4->10 (30) -0.8149955467976521, 4->11 (31) 3.183948191196622, 5->8 (32) -0.7770233692959215, 5->9 (33) -2.7242911816519566, 5->10 (34) -2.3552366031330836, 5->11 (35) 7.059413302232388, 6->8 (36) -1.6100837930348875, 6->9 (37) -0.0725816078919698, 6->10 (38) -6.516991035609596, 6->11 (39) -1.317628155709857, , 7->9 (41) -4.101860634518092, 7->10 (42) 5.291230053037152, 7->11 (43) -4.040944738052901, 6->143 (144) 1.5772684527260097, , 7->80 (81) 0.8467972668186209, 80->8 (82) 0.5279446224977301, 3->59 (60) 7.899622129310474, 59->10 (61) -1.9024018382919634, 0->143 (151) -5.589411876146301, 0->89 (90) -1.521222875392441, 89->10 (91) -0.9244674835827649, 143->89 (257) 0.8933547072045536, 80->143 (172) 2.5373617959793444, 5->89 (267) 1.6329092823436924, 3->120 (121) 1.5821302495347656, 120->9 (122) 4.356940472567296, 1->80 (198) -0.5022734576585477"

    client = LunarLanderClient()

    genotype = neat_tools.parse_genotype(nodes, edges)
    neat_tools.visualize_genotype(genotype)
    ann = Ann(genotype)

    client.render(ann)

    scores = []

    for i in range(1000):
        if i % 10 == 0:
            print(i)

        scores.append(client.get_fitness(ann))

    fitness = sum(scores) / len(scores)
    print(fitness)


if __name__ == "__main__":
    main()
