from dataset.dataset_xor import DatasetXor
from neat.neat import Neat

c1 = 1.0
c2 = 1.0
c3 = 0.4
t = 3
population_size = 150
generations = 200

#           Without Bias       ; With Bias
runs = 0  # 2000,              ; 1000
w_9 = 0   # 9                  ; 1
b_11 = 0  # 1977               ; 994
b_13 = 0  # 1945               ; 978
b_15 = 0  # 1829               ; 917
nodes = []  # 6.39     3.39    ; 6.358      2.358
edges = []  # 12.053   10.053  ; 11.971     8.971

# [0, 9] :  9    (0.45%)       ; 1   (0.1%)
# [9, 11]:  14   (0.7%)        ; 5   (0.5%)
# [11, 13]: 32   (1.6%)        ; 16  (1.6%)
# [13, 15]: 116  (5.8%)        ; 61  (6.1%)
# [15, 16]: 1829 (91.45%)      ; 917 (91.7%)

while True:
    runs += 1

    neat = Neat(c1, c2, c3, t, population_size, DatasetXor())
    neat.next_generations(generations)
    genotype = neat.get_best_genotype()

    if genotype.fitness <= 9:
        w_9 += 1

    if genotype.fitness > 11:
        b_11 += 1

    if genotype.fitness > 13:
        b_13 += 1

    if genotype.fitness > 15:
        b_15 += 1

    nodes.append(len(genotype.nodes))
    edges.append(len([e for e in genotype.edges if e.enabled]))

    print("Runs: " + str(runs) + ";\t\t11+: " + str(b_11) + ";\t\t13+: " + str(b_13) + ";\t\t15+: " + str(b_15) + "\t\t9-: " + str(w_9) + ";\t\tN: " + str(sum(nodes) / runs) + ";\t\tE: " + str(sum(edges) / runs))
