import gym

from gym_client.lunar_lander_client import LunarLanderClient
from neat.ann.ann import Ann
from neat.encoding.edge import Edge
from neat.encoding.genotype import Genotype
from neat.encoding.node import Node
from neat.encoding.node_type import NodeType

# 114.34978188221031 1000x
# nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.INPUT), Node(5, NodeType.INPUT), Node(6, NodeType.INPUT), Node(7, NodeType.INPUT), Node(8, NodeType.INPUT), Node(9, NodeType.OUTPUT), Node(10, NodeType.OUTPUT), Node(11, NodeType.OUTPUT), Node(12, NodeType.OUTPUT), Node(67, NodeType.HIDDEN), Node(167, NodeType.HIDDEN), Node(118, NodeType.HIDDEN), Node(76, NodeType.HIDDEN), Node(405, NodeType.HIDDEN), Node(471, NodeType.HIDDEN), Node(461, NodeType.HIDDEN), Node(869, NodeType.HIDDEN), Node(899, NodeType.HIDDEN), Node(350, NodeType.HIDDEN), Node(1163, NodeType.HIDDEN), Node(1241, NodeType.HIDDEN), Node(147, NodeType.HIDDEN), Node(657, NodeType.HIDDEN), Node(129, NodeType.HIDDEN), Node(1595, NodeType.HIDDEN), Node(745, NodeType.HIDDEN), Node(576, NodeType.HIDDEN), Node(773, NodeType.HIDDEN), Node(1684, NodeType.HIDDEN), Node(1146, NodeType.HIDDEN), Node(124, NodeType.HIDDEN), Node(1299, NodeType.HIDDEN), Node(1169, NodeType.HIDDEN), Node(2135, NodeType.HIDDEN)"
# edges = "0->9 (13) -0.06117504872860621, 0->10 (14) 0.08682454851442045, 0->11 (15) 0.07322899902642369, 0->12 (16) -0.01527213681658621, 1->9 (17) -0.01643869805770123, 1->10 (18) 0.12418105382948287, 1->11 (19) -0.1967685705842468, 1->12 (20) -0.017704278915546365, 2->9 (21) -0.1970473258497554, 2->10 (22) 0.13587657488312713, 2->11 (23) -0.2785579170465188, 2->12 (24) -0.18040251520784684, 3->9 (25) -0.01667339142561709, 3->10 (26) 0.12981350767782135, 3->11 (27) 0.185620021305832, 3->12 (28) -0.11291148769086364, 4->9 (29) -0.02012496330476754, 4->10 (30) -0.295502455281907, 4->11 (31) 0.04819181510680773, 4->12 (32) 0.11364239682062285, 5->9 (33) -0.09736929648836723, 5->10 (34) 0.03114984352086967, 5->11 (35) -0.18320497747823355, 5->12 (36) 0.047814153234324044, 6->9 (37) 0.05439475555180699, 6->10 (38) 0.08987308185726281, 6->11 (39) -0.5187488769023019, 6->12 (40) -0.21817943227814712, 7->9 (41) -0.3010046477953332, 7->10 (42) -0.060335283020307634, 7->11 (43) -0.15564192866744064, 7->12 (44) 0.08023538809885004, 8->9 (45) -0.01764923542175366, 8->10 (46) 0.21752826637367284, 8->11 (47) -0.03369898010973332, 8->12 (48) 0.055619327016617315, 7->67 (68) 1.089968714534041, 67->9 (69) 2.9435985056460288, 0->67 (195) -0.47735833148817136, 5->167 (168) 0.18895285798056174, 167->10 (169) 0.9504895982906797, 67->11 (206) 1.2817970120890374, 4->118 (119) 2.7836114975498725, 118->11 (120) -0.5641594650926862, 67->10 (185) 0.19423477304057585, 5->76 (77) 1.435656162857051, 76->12 (78) -0.017760327726894478, 118->10 (344) -1.0045948336651183, 0->405 (406) -0.130918589019132, 405->67 (407) -1.3116746021280905, 1->167 (459) 3.0175370127705756, 5->471 (472) 0.906753300623581, 471->76 (473) 1.3483726800433482, 5->461 (462) 0.8800449053463901, 461->10 (463) -0.7475479938844909, 3->167 (633) -0.35193007989836933, 8->471 (528) 1.9688197110823413, 7->167 (599) 0.35116833238664136, 4->67 (245) -1.064468111676057, 6->471 (672) 1.1937000794716102, 6->869 (870) 0.6494302612950552, 869->471 (871) 1.6234771856081003, 67->12 (223) -1.222749175679633, 869->11 (958) -0.8525854021111702, 118->76 (429) 0.7395552212088439, 471->11 (649) -0.3172626672666499, 8->869 (915) -2.5002067501280343, 4->76 (536) 0.06636464649436283, 3->899 (900) 2.2384416271285223, 899->167 (901) 0.07879202987757622, 5->350 (351) 0.4249726660220093, 350->167 (352) 0.4188635562166129, 4->471 (708) 0.012242725191133884, 7->869 (1061) 0.6635518223001983, 405->869 (957) 0.7419462688932078, 3->1163 (1164) 2.375951754003231, 1163->899 (1165) 1.088117519868004, 869->10 (960) 0.8108691413673043, 3->1241 (1242) 0.5586864011925299, 1241->899 (1243) 2.979480368313708, 118->471 (618) -0.06270281950395643, 3->147 (148) -0.2521420204791796, 147->11 (149) 1.4495376458056173, 461->899 (1160) 0.22971966166920865, 7->147 (277) -0.06490620094264982, 118->657 (658) 1.761255355752627, 657->471 (659) 0.804599795378354, 0->147 (387) 0.4185483592397494, 67->899 (1159) 0.9178804946099954, 3->129 (130) 1.3425768472645152, 129->10 (131) -0.16142257690615025, 67->1595 (1596) 2.448351408819929, 1595->899 (1597) 0.6475487328199764, 657->9 (1719) -1.2788128689731357, 3->657 (1761) 0.14627328965101039, 5->745 (746) 0.8148509574910575, 745->471 (747) 1.134837604886078, 1241->657 (1511) -1.2491670963590065, 1595->167 (1808) 0.35927970783362295, 3->461 (858) -1.3254529868906608, 1->869 (1084) -1.5887170768429684, 1241->11 (1257) -0.18185742340418065, 6->67 (216) -0.03752189745453244, 2->67 (239) 0.6376849372351754, 118->576 (577) 1.4352792365538911, 576->76 (578) 0.07538078915606808, 5->657 (1526) 0.3254028972920125, 129->471 (783) 0.6506173866358805, 5->773 (774) 1.5350881895117539, 773->350 (775) 0.2685263952488307, 461->9 (736) -0.5473248929741451, 118->745 (1915) 0.39907344277969614, 147->461 (1445) -0.6524034479935734, 5->1684 (1685) -0.4232035085755363, 1684->773 (1686) 0.24774867456088107, 3->1146 (1147) 0.6755661284532236, 1146->167 (1148) -0.07808447680780282, 869->657 (1573) 1.4598150398842835, 1->745 (1828) -0.6600861840412136, 7->76 (177) -0.7130988899042794, 461->1241 (1376) -0.18496796991593556, 745->129 (1846) 0.04767900917970674, 2->1241 (1340) -0.7381616802903878, 471->1684 (2258) -2.0784642463436342, 4->773 (2272) 0.6842220894844557, 1->124 (125) 0.7740329487922765, 124->9 (126) -0.040962035624346645, 1684->1146 (2359) -0.389347239062586, 6->76 (391) 1.999483644225771, 147->1595 (1760) -3.3052395308624476, 3->1299 (1300) 1.96664106579073, 1299->1241 (1301) 1.0810991841030826, 7->1684 (2478) 2.786068821506742, 8->1169 (1170) -0.8500107704589038, 1169->869 (1171) -2.427970512064406, 2->1163 (1466) 0.6843523406333062, 129->2135 (2136) 0.445737954004679, 2135->471 (2137) 0.8553212844001863, 8->1146 (2562) 1.8130212984159737, 8->745 (1823) 0.3650979582260576, 4->405 (614) -0.5386586451465046, 461->773 (2194) 0.4455908956699136, 1595->129 (1728) 0.02703786174473767"

# 150.11502059468333 1000x
# nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.INPUT), Node(5, NodeType.INPUT), Node(6, NodeType.INPUT), Node(7, NodeType.INPUT), Node(8, NodeType.INPUT), Node(9, NodeType.OUTPUT), Node(10, NodeType.OUTPUT), Node(11, NodeType.OUTPUT), Node(12, NodeType.OUTPUT), Node(67, NodeType.HIDDEN), Node(167, NodeType.HIDDEN), Node(118, NodeType.HIDDEN), Node(76, NodeType.HIDDEN), Node(405, NodeType.HIDDEN), Node(471, NodeType.HIDDEN), Node(461, NodeType.HIDDEN), Node(869, NodeType.HIDDEN), Node(899, NodeType.HIDDEN), Node(350, NodeType.HIDDEN), Node(1163, NodeType.HIDDEN), Node(1241, NodeType.HIDDEN), Node(147, NodeType.HIDDEN), Node(657, NodeType.HIDDEN), Node(129, NodeType.HIDDEN), Node(1595, NodeType.HIDDEN), Node(745, NodeType.HIDDEN), Node(576, NodeType.HIDDEN), Node(773, NodeType.HIDDEN), Node(1684, NodeType.HIDDEN), Node(1146, NodeType.HIDDEN), Node(124, NodeType.HIDDEN), Node(1299, NodeType.HIDDEN), Node(1169, NodeType.HIDDEN), Node(2135, NodeType.HIDDEN)"
# edges = "0->9 (13) -0.0611750487286062, 0->10 (14) 0.20868843967208572, 0->11 (15) 0.07322899902642369, 0->12 (16) -0.11014226128439958, 1->9 (17) -0.016438698057701227, 1->10 (18) 0.12418105382948287, 1->11 (19) -0.19676857058424668, 1->12 (20) -0.017704278915546365, 2->9 (21) -0.19704732584975548, 2->10 (22) 0.13587657488312704, 2->11 (23) -0.2785579170465188, 2->12 (24) -0.18040251520784684, 3->9 (25) -0.016673391425617086, 3->10 (26) 0.1415809314998149, 3->11 (27) 0.1856200213058319, 3->12 (28) -0.11291148769086364, 4->9 (29) 0.008089578773127597, 4->10 (30) -0.295502455281907, 4->11 (31) 0.04819181510680775, 4->12 (32) 0.11364239682062285, 5->9 (33) -0.09736929648836723, 5->10 (34) 0.03114984352086967, 5->11 (35) -0.2119751274472013, 5->12 (36) 0.047814153234324044, 6->9 (37) 0.05439475555180699, 6->10 (38) 0.05680235769930022, 6->11 (39) -0.02548885038451003, 6->12 (40) -0.21817943227814718, 7->9 (41) -0.2572282551429827, 7->10 (42) -0.060335283020307634, 7->11 (43) -0.14916968121666513, 7->12 (44) 0.07349840660569716, 8->9 (45) -0.017394127462943693, 8->10 (46) 0.22438889650318275, 8->11 (47) -0.0336989801097333, 8->12 (48) 0.05572782265472923, 7->67 (68) 1.0631464120665335, 67->9 (69) 2.9435985056460288, 0->67 (195) -0.4773583314881714, 5->167 (168) 0.18895285798056172, 167->10 (169) 0.9504895982906796, 67->11 (206) 1.2817970120890374, 4->118 (119) 2.7697689805249, 118->11 (120) -0.6268434308249878, 67->10 (185) 0.19423477304057585, 5->76 (77) 1.435656162857051, 76->12 (78) -0.019428233841798846, 118->10 (344) -1.0045948336651191, 0->405 (406) -0.130918589019132, 405->67 (407) -1.3199495289719156, 1->167 (459) 3.0175370127705756, 5->471 (472) 1.0698242634648323, 471->76 (473) 1.3597319705140058, 5->461 (462) 0.8800449053463901, 461->10 (463) -0.7475479938844909, 3->167 (633) -0.35193007989836933, 8->471 (528) 1.968819711082342, 7->167 (599) 0.35116833238664136, 4->67 (245) -1.0644681116760577, 6->471 (672) 1.1937000794716104, 6->869 (870) 0.6494302612950553, 869->471 (871) 1.6234771856081, 67->12 (223) -1.222749175679633, 869->11 (958) -0.8525854021111701, 118->76 (429) 0.7395552212088443, 471->11 (649) -0.3172626672666509, 8->869 (915) -2.5002067501280334, 4->76 (536) 0.0663646464943628, 3->899 (900) 2.2384416271291268, 899->167 (901) 0.07879202987769963, 5->350 (351) 0.42497266602200956, 350->167 (352) 0.41886355621661303, 4->471 (708) 0.01224272519102327, 7->869 (1061) 0.6635518223001771, 405->869 (957) 0.7419462688931947, 3->1163 (1164) 2.37595175400323, 1163->899 (1165) 1.0881175198680042, 869->10 (960) 0.8108691413673123, 3->1241 (1242) 0.5586864011925348, 1241->899 (1243) 2.979480368313708, 118->471 (618) -0.06270281950369994, 3->147 (148) -0.2521420204783114, 147->11 (149) 1.4495376458022537, 461->899 (1160) 0.2297196616692248, 7->147 (277) -0.06490620094281649, 118->657 (658) 1.761255355768894, 657->471 (659) 0.804599795379142, 0->147 (387) 0.4185483592392, 67->899 (1159) 0.9178804946084526, 3->129 (130) 1.3425768472716622, 129->10 (131) -0.16142257690651712, 67->1595 (1596) 2.4483514088066003, 1595->899 (1597) 0.6475487328191883, 657->9 (1719) -1.2788128689752771, 3->657 (1761) 0.1462732892876289, 5->745 (746) 0.8148509580582343, 745->471 (747) 1.1348376042382373, 1241->657 (1511) -1.2491670964220125, 1595->167 (1808) 0.3592797059705046, 3->461 (858) -1.3254529851245873, 1->869 (1084) -1.5887170768059293, 1241->11 (1257) -0.18185742178138445, 6->67 (216) -0.03752189389074381, 2->67 (239) 0.6376849265051766, 118->576 (577) 1.4352792143073112, 576->76 (578) 0.07538080250274716, 5->657 (1526) 0.3254028379247399, 129->471 (783) 0.6506173372436935, 5->773 (774) 1.535088221809006, 773->350 (775) 0.26852639321258237, 461->9 (736) -0.5473246506332006, 118->745 (1915) 0.3990735527943978, 147->461 (1445) -0.6524034379819101, 5->1684 (1685) -0.42324117337659845, 1684->773 (1686) 0.24775419853109862, 3->1146 (1147) 0.6755709523299603, 1146->167 (1148) -0.07808431701956928, 869->657 (1573) 1.4598153303136039, 1->745 (1828) -0.6600847190294235, 7->76 (177) -0.7130996677958023, , 745->129 (1846) 0.04767825017777726, 2->1241 (1340) -0.7381421136649523, 471->1684 (2258) -2.0785368396865342, 4->773 (2272) 0.6843135997704493, 1->124 (125) 0.7740460993052665, 124->9 (126) -0.0409676941638739, 1684->1146 (2359) -0.3881175945816187, 6->76 (391) 2.00407721433343, 147->1595 (1760) -3.302236774492962, 3->1299 (1300) 1.9665434139614528, 1299->1241 (1301) 1.0778667598506677, 7->1684 (2478) 2.7982784036657846, 8->1169 (1170) -0.8544255358647946, 1169->869 (1171) -2.4344727109190547, 2->1163 (1466) 0.684234170472573, 129->2135 (2136) 0.4392326753549535, 2135->471 (2137) 0.8383330515002532, 8->1146 (2562) 1.8376368621014825, 8->745 (1823) 0.3446632815063826, 4->405 (614) -0.6660778977770867, 1169->67 (2683) -1.270452385669422, 461->167 (875) -0.6293932506697736, 8->2135 (2846) -0.1811851625424909"

# 186.09892090720376 1000x
nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.INPUT), Node(5, NodeType.INPUT), Node(6, NodeType.INPUT), Node(7, NodeType.INPUT), Node(8, NodeType.INPUT), Node(9, NodeType.OUTPUT), Node(10, NodeType.OUTPUT), Node(11, NodeType.OUTPUT), Node(12, NodeType.OUTPUT), Node(67, NodeType.HIDDEN), Node(167, NodeType.HIDDEN), Node(118, NodeType.HIDDEN), Node(76, NodeType.HIDDEN), Node(405, NodeType.HIDDEN), Node(471, NodeType.HIDDEN), Node(461, NodeType.HIDDEN), Node(869, NodeType.HIDDEN), Node(899, NodeType.HIDDEN), Node(350, NodeType.HIDDEN), Node(1163, NodeType.HIDDEN), Node(1241, NodeType.HIDDEN), Node(147, NodeType.HIDDEN), Node(657, NodeType.HIDDEN), Node(129, NodeType.HIDDEN), Node(1595, NodeType.HIDDEN), Node(745, NodeType.HIDDEN), Node(576, NodeType.HIDDEN), Node(773, NodeType.HIDDEN), Node(1684, NodeType.HIDDEN), Node(1146, NodeType.HIDDEN), Node(124, NodeType.HIDDEN), Node(1299, NodeType.HIDDEN), Node(1169, NodeType.HIDDEN), Node(2135, NodeType.HIDDEN), Node(2053, NodeType.HIDDEN), Node(2226, NodeType.HIDDEN), Node(2585, NodeType.HIDDEN), Node(334, NodeType.HIDDEN), Node(548, NodeType.HIDDEN), Node(1838, NodeType.HIDDEN), Node(1984, NodeType.HIDDEN), Node(2708, NodeType.HIDDEN), Node(2427, NodeType.HIDDEN), Node(627, NodeType.HIDDEN), Node(3574, NodeType.HIDDEN), Node(1830, NodeType.HIDDEN)"
edges = "0->9 (13) -0.06117504872860623, 0->10 (14) 0.17386683669925812, 0->11 (15) 0.16745637957632897, 0->12 (16) -0.11821758637831394, 1->9 (17) -0.03925918299405816, 1->10 (18) 0.11553306257377546, 1->11 (19) -0.19159162560669102, 1->12 (20) 0.05506930175455718, 2->9 (21) -0.19704732584975548, 2->10 (22) 0.28910404639543, 2->11 (23) -0.21409017368900307, 2->12 (24) -0.32528056670760847, 3->9 (25) 0.00112261026527878, 3->10 (26) 0.13914674851098427, 3->11 (27) -0.025183031247289923, 3->12 (28) 0.05319548370063097, 4->9 (29) 0.005652444452273766, 4->10 (30) -0.47645547492218815, 4->11 (31) 0.04819181510680775, 4->12 (32) 0.26604560646505215, 5->9 (33) -0.09736929648836723, 5->10 (34) -0.020797378246074222, 5->11 (35) -0.21048745272168692, 5->12 (36) 0.1809493565706431, 6->9 (37) 0.1397583991852555, 6->10 (38) 0.18000167602387535, 6->11 (39) 0.05405814950535602, 6->12 (40) -0.21817943227814718, 7->9 (41) -0.26436441059444726, 7->10 (42) 0.05775450933626207, 7->11 (43) -0.14479333828881452, 7->12 (44) 0.07259025077303786, 8->9 (45) -0.01727807227655358, 8->10 (46) 0.22448122139565563, 8->11 (47) -0.09651134536053321, 8->12 (48) 0.055708370453405306, 7->67 (68) 1.0609462257447722, 67->9 (69) 3.045309626877274, 0->67 (195) -0.4773583314881713, 5->167 (168) 0.34906080464891237, 167->10 (169) 0.9504895982906796, 67->11 (206) 1.2817970120890378, 4->118 (119) 2.77404515161595, 118->11 (120) -0.62468071167304, 67->10 (185) 0.33962934368523096, 5->76 (77) 1.4092128223474627, 76->12 (78) -0.02385858785791724, 118->10 (344) -1.1598757552270147, 0->405 (406) -0.17857732090081613, 405->67 (407) -1.3262548648560815, 1->167 (459) 3.0802683555005745, 5->471 (472) 1.0619359955646661, 471->76 (473) 1.387372805766454, 5->461 (462) 0.8426606469415009, 461->10 (463) -0.7475479938844909, 3->167 (633) -0.3519300798983693, 8->471 (528) 1.8258148757486372, 7->167 (599) 0.3140016631511837, 4->67 (245) -1.0235168184278869, 6->471 (672) 1.2027588283282902, 6->869 (870) 0.6115596239984223, 869->471 (871) 1.6234771856081005, 67->12 (223) -1.222749175679633, 869->11 (958) -0.8525854021111703, 118->76 (429) 1.1360716298758955, 471->11 (649) -0.31726266726665076, 8->869 (915) -2.5002067501280347, 4->76 (536) 0.06636464649436274, 3->899 (900) 2.238441627129093, 899->167 (901) 0.07879202987779528, 5->350 (351) 0.4249726660220097, 350->167 (352) 0.4188635562166129, 4->471 (708) 0.012242725191037009, 7->869 (1061) 0.6843500707723216, 405->869 (957) 0.7419462688931893, 3->1163 (1164) 2.2754541794173955, 1163->899 (1165) 1.0881175198680049, 869->10 (960) 0.8108691413673126, 3->1241 (1242) 0.5586864011925352, 1241->899 (1243) 2.979480368313708, 118->471 (618) -0.06270281950367496, 3->147 (148) -0.252142020478312, 147->11 (149) 1.449537645802534, 461->899 (1160) 0.2297196616691669, 7->147 (277) -0.06490620094306643, 118->657 (658) 1.7612553557688735, 657->471 (659) 0.80459979537897, 0->147 (387) 0.4185483592391831, 67->899 (1159) 0.9178804946080308, 3->129 (130) 1.3425768472720763, 129->10 (131) -0.1614225769064615, 67->1595 (1596) 2.4483514088035276, 1595->899 (1597) 0.6475487328192804, 657->9 (1719) -1.278812868975702, 3->657 (1761) 0.14627328950252153, 5->745 (746) 0.8148509580435862, 745->471 (747) 1.1348376042635655, 1241->657 (1511) -1.2491670965534223, 1595->167 (1808) 0.35927970598596426, 3->461 (858) -1.3254529853608361, 1->869 (1084) -1.5887170768237207, 1241->11 (1257) -0.18185742233664334, 6->67 (216) -0.03752189382485611, 2->67 (239) 0.6376849270618937, 118->576 (577) 1.4352792120950832, 576->76 (578) 0.07538080347257041, 5->657 (1526) 0.3254028472310318, 129->471 (783) 0.6506173397145386, 5->773 (774) 1.5350882229433507, 773->350 (775) 0.2685263948546733, 461->9 (736) -0.5473246501957644, 118->745 (1915) 0.39907354108933224, 147->461 (1445) -0.6524034387765074, 5->1684 (1685) -0.42323210751196405, 1684->773 (1686) 0.24775423728656204, 3->1146 (1147) 0.6755716243898112, 1146->167 (1148) -0.0780849902237546, 869->657 (1573) 1.4598153040615416, 1->745 (1828) -0.6600843925874422, 7->76 (177) -0.7130997092438102, 461->1241 (1376) -0.18496816819323247, 745->129 (1846) 0.04767808331857727, 2->1241 (1340) -0.7381292861757742, 471->1684 (2258) -2.0785248156580605, 4->773 (2272) 0.6843103060229752, 1->124 (125) 0.7740483419393305, 124->9 (126) -0.040967367832945495, 1684->1146 (2359) -0.3881790628573389, 6->76 (391) 2.003427444117967, 147->1595 (1760) -3.3010316901240517, 3->1299 (1300) 1.9667493989536038, 1299->1241 (1301) 1.0772585663697345, 7->1684 (2478) 2.8020538374746815, 8->1169 (1170) -0.8557397510591305, 1169->869 (1171) -2.434254468619185, 2->1163 (1466) 0.684708539940906, 129->2135 (2136) 0.4393137243259293, 2135->471 (2137) 0.836775638457001, 8->1146 (2562) 1.8295132466652895, 8->745 (1823) 0.3383717700582517, 4->405 (614) -0.6479104660778108, 1169->67 (2683) -1.277596000403689, 773->2053 (2054) 0.8884379063291885, 2053->350 (2055) -0.5043962243827811, 869->2226 (2227) 0.7305404564917886, 2226->657 (2228) 0.7859713165154184, 2135->576 (2602) 0.026578169878933543, 1169->899 (2652) 0.4179281669448993, 6->129 (1149) 0.4342731188200963, 147->2585 (2586) 1.4735405045217704, 2585->1595 (2587) -3.672642944533697, 5->2135 (2880) -0.29957187654355755, 124->1299 (2687) 0.14182925788406028, 1595->76 (1767) 0.5212845062535936, 76->350 (630) -0.8114702287854954, 869->147 (1468) -1.662639075895973, 1169->405 (1520) 0.504397842004616, 67->334 (335) 1.2275095063309005, 334->10 (336) -0.31280457898782754, 745->2053 (2944) -0.056873409111568934, 405->548 (549) -0.9951560771240373, 548->67 (550) 1.0461980300311222, 118->869 (1001) -0.9549975950473177, 3->1838 (1839) 1.43128450220073, 1838->1146 (1840) 0.9703275092521599, 0->471 (613) -0.7973608250475477, 67->1146 (2071) 0.8740331356009878, 3->334 (3127) -1.6115452029784052, 5->2226 (3218) 0.44242531196143453, 7->899 (1618) -1.2597797801225168, 1241->1984 (1985) 1.1010232919051952, 1984->657 (1986) -1.1782491707804126, 8->2708 (2709) 0.9236695123148111, 2708->745 (2710) 0.9158170463858297, 1684->2427 (2428) 1.1192871389944772, 2427->1146 (2429) -1.032900308546103, 8->899 (1189) 0.04668679537367526, 1163->2053 (2557) 2.068519392856494, 5->627 (628) 3.0056185271915252, 627->76 (629) -0.6670748310230064, 3->3574 (3575) 0.3780875359619676, 3574->334 (3576) -2.742564589215221, 461->1830 (1832) 0.3805256759529644, 1830->1241 (1831) 0.27197219767437475"

# 248.56512261124243 1000x
nodes = "Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.INPUT), Node(3, NodeType.INPUT), Node(4, NodeType.INPUT), Node(5, NodeType.INPUT), Node(6, NodeType.INPUT), Node(7, NodeType.INPUT), Node(8, NodeType.INPUT), Node(9, NodeType.OUTPUT), Node(10, NodeType.OUTPUT), Node(11, NodeType.OUTPUT), Node(12, NodeType.OUTPUT), Node(67, NodeType.HIDDEN), Node(167, NodeType.HIDDEN), Node(118, NodeType.HIDDEN), Node(76, NodeType.HIDDEN), Node(405, NodeType.HIDDEN), Node(471, NodeType.HIDDEN), Node(461, NodeType.HIDDEN), Node(869, NodeType.HIDDEN), Node(899, NodeType.HIDDEN), Node(350, NodeType.HIDDEN), Node(1163, NodeType.HIDDEN), Node(1241, NodeType.HIDDEN), Node(147, NodeType.HIDDEN), Node(657, NodeType.HIDDEN), Node(129, NodeType.HIDDEN), Node(1595, NodeType.HIDDEN), Node(745, NodeType.HIDDEN), Node(576, NodeType.HIDDEN), Node(773, NodeType.HIDDEN), Node(1684, NodeType.HIDDEN), Node(1146, NodeType.HIDDEN), Node(124, NodeType.HIDDEN), Node(1299, NodeType.HIDDEN), Node(1169, NodeType.HIDDEN), Node(2135, NodeType.HIDDEN), Node(2053, NodeType.HIDDEN), Node(2226, NodeType.HIDDEN), Node(2585, NodeType.HIDDEN), Node(334, NodeType.HIDDEN), Node(548, NodeType.HIDDEN), Node(1838, NodeType.HIDDEN), Node(1984, NodeType.HIDDEN), Node(2708, NodeType.HIDDEN), Node(2427, NodeType.HIDDEN), Node(627, NodeType.HIDDEN), Node(199, NodeType.HIDDEN), Node(2309, NodeType.HIDDEN)"
edges = "0->9 (13) 0.4812248878716398, 0->10 (14) 0.17386706995891943, 0->11 (15) 0.16773798214426475, 0->12 (16) -0.1182175658205114, 1->9 (17) -0.03929509079142421, 1->10 (18) 0.11461418721419256, 1->11 (19) -0.19159090059791736, 1->12 (20) 0.055069907394839426, 2->9 (21) -0.19704732584975548, 2->10 (22) 0.28560413743389335, 2->11 (23) -0.21520461083012718, 2->12 (24) -0.3820799425808404, 3->9 (25) 0.0011393668489576177, 3->10 (26) 0.13914674786941203, 3->11 (27) -0.02513891617724573, 3->12 (28) 0.04200734065997739, 4->9 (29) 0.005652443705178288, 4->10 (30) -0.47646844497677854, 4->11 (31) 0.04819181510680773, 4->12 (32) 0.2660462015943028, 5->9 (33) -0.09736929648836723, 5->10 (34) -0.02080036255305082, 5->11 (35) -0.21048758587081914, 5->12 (36) 0.18094907788753967, 6->9 (37) 0.13975732859230722, 6->10 (38) 0.1800020813372675, 6->11 (39) 0.0540581319161847, 6->12 (40) -0.21817943227814712, 7->9 (41) -0.2643644141066997, 7->10 (42) 0.05775451831735613, 7->11 (43) -0.14479338356727062, 7->12 (44) 0.07259024701998738, 8->9 (45) -0.017278072463051602, 8->10 (46) 0.224481221599949, 8->11 (47) -0.09651031483502255, 8->12 (48) 0.05570837044364173, 7->67 (68) 1.0609462170964619, 67->9 (69) 3.0451212581573692, 0->67 (195) -0.4773583314881713, 5->167 (168) 0.3492022230559827, 167->10 (169) 0.9504895982906796, 67->11 (206) 1.2817970120890374, 4->118 (119) 2.774045149076355, 118->11 (120) -0.6246807084714635, 67->10 (185) 0.33963227991307815, 5->76 (77) 1.4091413250351197, 76->12 (78) -0.023858589176406788, 118->10 (344) -1.1635359034350754, 0->405 (406) -0.17856874463507644, 405->67 (407) -1.3262551809990537, 1->167 (459) 3.08027025490053, 5->471 (472) 1.0619358614396366, 471->76 (473) 1.3873727761120191, 5->461 (462) 0.8457813765408797, 461->10 (463) -0.7475479938844909, 3->167 (633) -0.35193007989836933, 8->471 (528) 1.8289477802303649, 7->167 (599) 0.31423284531547924, 4->67 (245) -1.03379441055039, 6->471 (672) 1.2287959919304132, 6->869 (870) 0.6115581694536388, 869->471 (871) 1.6234771856081005, 67->12 (223) -1.2227491756796334, 869->11 (958) -0.8525854021111703, 118->76 (429) 0.9037951697505711, 471->11 (649) -0.317262667266651, 8->869 (915) -2.5002067501280347, 4->76 (536) 0.06636464649436276, 3->899 (900) 2.238441627129093, 899->167 (901) 0.07879202987779528, 5->350 (351) 0.42497266602200967, 350->167 (352) 0.2804890602444187, 4->471 (708) 0.012242725191037012, 7->869 (1061) 0.6839959204191836, 405->869 (957) 0.4521797355315065, 3->1163 (1164) 2.2733745963348615, 1163->899 (1165) 1.088117519868005, 869->10 (960) 0.8108691413673126, 3->1241 (1242) 0.5586864011925348, 1241->899 (1243) 2.9794803683137077, 118->471 (618) -0.06270281950367496, 3->147 (148) -0.252142020478312, 147->11 (149) 1.449537645802534, 461->899 (1160) 0.2297196616691669, 7->147 (277) -0.2727389909595985, 118->657 (658) 1.7612553557688733, 657->471 (659) 0.8045997953789701, 0->147 (387) 0.418548359239183, 67->899 (1159) 0.9178804946080308, 3->129 (130) 1.3425768472720758, 129->10 (131) -0.1614225769064615, 67->1595 (1596) 2.4483514088035276, 1595->899 (1597) 0.6475487328192804, 657->9 (1719) -1.2788128689757023, 3->657 (1761) 0.14627328950252153, 5->745 (746) 0.8148509580435863, 745->471 (747) 1.134837604263565, 1241->657 (1511) -1.249167096553423, 1595->167 (1808) 0.35927970598596415, 3->461 (858) -1.3254529853608363, 1->869 (1084) -1.5887170768237207, 1241->11 (1257) -0.18185742233664304, 6->67 (216) -0.03752189382485677, 2->67 (239) 0.6376849270618947, 118->576 (577) 1.4352792120950855, 576->76 (578) 0.07538080347257195, 5->657 (1526) 0.32540284723102547, 129->471 (783) 0.6506173397145427, 5->773 (774) 1.5350882229433496, 773->350 (775) 0.26852639485467045, 461->9 (736) -0.5473246501957866, 118->745 (1915) 0.39907354108937904, 147->461 (1445) -0.6524034387765073, 5->1684 (1685) -0.4232321075212466, 1684->773 (1686) 0.2477542372880909, 3->1146 (1147) 0.6755716243881552, 1146->167 (1148) -0.07808499022603672, 869->657 (1573) 1.4598153040615338, 1->745 (1828) -0.6600843925869764, 7->76 (177) -0.7130997092441982, 461->1241 (1376) -0.18496816819544232, 745->129 (1846) 0.04767808331964331, 2->1241 (1340) -0.7381292861290358, 471->1684 (2258) -2.0785248155109177, 4->773 (2272) 0.6843103060062008, 1->124 (125) 0.7740483419949407, 124->9 (126) -0.04096736783203627, 1684->1146 (2359) -0.3881790628499773, 6->76 (391) 2.0034274448750984, 147->1595 (1760) -3.3010316903490873, 3->1299 (1300) 1.966749398640101, 1299->1241 (1301) 1.0772585656144675, 7->1684 (2478) 2.802053827837785, 8->1169 (1170) -0.8557397513574095, 1169->869 (1171) -2.4342544698367443, 2->1163 (1466) 0.6847085388817076, 129->2135 (2136) 0.4393137239789285, 2135->471 (2137) 0.8367756357020548, 8->1146 (2562) 1.8295132530671976, 8->745 (1823) 0.3383717836577018, 4->405 (614) -0.6479105075665548, 1169->67 (2683) -1.2775960813811318, 773->2053 (2054) 0.8884379390541544, 2053->350 (2055) -0.5043962150145664, 869->2226 (2227) 0.7305602537617547, 2226->657 (2228) 0.7859704636952837, 2135->576 (2602) 0.026604057256124795, 1169->899 (2652) 0.4179283697859842, 6->129 (1149) 0.4342737730369884, 147->2585 (2586) 1.4735414332605108, 2585->1595 (2587) -3.6726430939042447, 5->2135 (2880) -0.2995769099408746, 124->1299 (2687) 0.14182887438995223, 1595->76 (1767) 0.5212661199656288, 76->350 (630) -0.8114730476286843, 869->147 (1468) -1.662649510088103, 1169->405 (1520) 0.504398781508458, 67->334 (335) 1.2275077766772922, 334->10 (336) -0.31280438196745813, 745->2053 (2944) -0.05687539086378489, 405->548 (549) -0.9951489738670072, 548->67 (550) 1.0461601047702898, 118->869 (1001) -0.9550077283755701, 3->1838 (1839) 1.431279538724238, 1838->1146 (1840) 0.9703206330560619, 0->471 (613) -0.7969441834627761, 67->1146 (2071) 0.8739904260560554, 3->334 (3127) -1.6396823248318806, 5->2226 (3218) 0.4509462917680251, 7->899 (1618) -1.2354973603483994, 1241->1984 (1985) 1.1103704543984758, 1984->657 (1986) -1.133393688891585, 8->2708 (2709) 0.9385420705445009, 2708->745 (2710) 0.9448508834581378, 1684->2427 (2428) 1.1049725482979873, 2427->1146 (2429) -1.0270114900617118, 8->899 (1189) 0.046155462084485135, 1163->2053 (2557) 1.907423803804848, 5->627 (628) 3.597198089186144, 627->76 (629) -0.5627799335784879, 0->199 (200) 1.6125214580702383, 199->10 (201) 0.39056279640262254, 350->334 (3608) 2.783477399052084, 745->2309 (2310) 1.1013904239512415, 2309->129 (2311) 0.3248459188000566, 2708->2585 (3522) 0.4973035495320875, 350->2427 (3477) -0.8066244574330655, 2708->129 (3539) 1.516661433974753, 199->471 (3692) 0.4894309381885988, 576->773 (2174) -0.8672062684321227, 1684->1163 (2167) 0.43799035678477916"


genotype = Genotype()

nodes = nodes.replace("Node", "").split(", ")

for i in range(0, len(nodes), 2):
    node_id = int(nodes[i].replace("(", ""))
    node_type = nodes[i+1].replace(")", "")

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

ann = Ann(genotype)
client = LunarLanderClient()

client.render(ann)

scores = []

for i in range(1000):
    if i % 10 == 0:
        print(i)

    scores.append(client.get_fitness(ann))

fitness = sum(scores) / len(scores)
print(fitness)
