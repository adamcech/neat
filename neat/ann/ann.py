from typing import List, Tuple, Dict, Callable
import math

from neat.ann.target_function import TargetFunction
from neat.encoding.node_activation import NodeActivation


class Ann:

    def __init__(self, genotype: "Genotype", target_function: TargetFunction):
        node_layers, layers_list = genotype.get_node_layers()

        self._layers = len(layers_list)
        self.target_function = target_function
        self._neurons = [[] for _ in range(self._layers)]  # type: List[List[float]]
        self._act = [[] for _ in range(self._layers)]  # type: List[List[Callable]]

        nodes_dict = {n.id: n for n in genotype.nodes}  # type: Dict[int, "Node"]
        nodes_id = {}  # type: Dict[int, Tuple[int, int]]
        for i in range(len(layers_list)):
            for neuron_id in layers_list[i]:

                nodes_id[neuron_id] = (i, len(self._neurons[i]))
                self._neurons[i].append(0)
                self._act[i].append(self.get_activation(nodes_dict[neuron_id].activation))

        # nodes_id = {}  # type: Dict[int, Tuple[int, int]]
        # for node in genotype.nodes:
        #     if node.is_output():
        #         continue
        #
        #     layer = node_layers[node.id]
        #
        #     nodes_id[node.id] = (layer, len(self._neurons[layer]))
        #     self._neurons[layer].append(0)
        #     self._act[layer].append(self.get_activation(node.activation))

        self._weights = [[[] for _ in layer] for layer in self._neurons]  # type: List[List[List[Tuple[float, int, int]]]]

        for edge in genotype.edges:
            if edge.enabled:
                input_layer, input_node = nodes_id[edge.input]
                output_layer, output_node = nodes_id[edge.output]
                self._weights[output_layer][output_node].append((edge.weight, input_layer, input_node))

    def calculate(self, item_input) -> List[float]:
        self._neurons[0] = item_input

        for neuron in range(len(self._neurons[0])):
            self._neurons[0][neuron] = float(item_input[neuron])

        for layer in range(1, len(self._neurons) - 1):
            for neuron in range(len(self._neurons[layer])):
                self._neurons[layer][neuron] = self._act[layer][neuron](sum(
                    self._neurons[input_layer][input_node] * weight for weight, input_layer, input_node in self._weights[layer][neuron]))

        for neuron in range(len(self._neurons[-1])):
            self._neurons[-1][neuron] = sum(self._neurons[input_layer][input_node] * weight for weight, input_layer, input_node in self._weights[-1][neuron])
        self.__calculate_target_function()

        return self._neurons[-1]

    def __calculate_target_function(self):
        if self.target_function == TargetFunction.SOFTMAX:
            self._neurons[-1] = self.__softmax(self._neurons[-1])
        elif self.target_function == TargetFunction.RELU:
            self._neurons[-1] = [self.__relu(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.SIGM:
            self._neurons[-1] = [self.__sigm(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.STEEPENED_SIGM:
            self._neurons[-1] = [self.__steepened_sigm(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.TANH:
            self._neurons[-1] = [self.__tanh(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.STEEPENED_TANH:
            self._neurons[-1] = [self.__steepened_tanh(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.CLAMPED:
            self._neurons[-1] = [self.__clamped(neuron) for neuron in self._neurons[-1]]
        elif self.target_function == TargetFunction.LIN:
            self._neurons[-1] = [self.__lin(neuron) for neuron in self._neurons[-1]]

    def get_activation(self, activation: NodeActivation) -> Callable:
        if activation == NodeActivation.CLAMPED:
            return self.__clamped
        elif activation == NodeActivation.SIGM:
            return self.__sigm
        elif activation == NodeActivation.STEEPENED_SIGM:
            return self.__steepened_sigm
        elif activation == NodeActivation.TANH:
            return self.__tanh
        elif activation == NodeActivation.STEEPENED_TANH:
            return self.__steepened_tanh
        elif activation == NodeActivation.RELU:
            return self.__relu
        elif activation == NodeActivation.INV:
            return self.__inv
        elif activation == NodeActivation.SIN:
            return self.__sin
        elif activation == NodeActivation.COS:
            return self.__cos
        elif activation == NodeActivation.ABS:
            return self.__abs
        elif activation == NodeActivation.STEP:
            return self.__step
        elif activation == NodeActivation.LIN:
            return self.__lin
        elif activation == NodeActivation.GAUSS:
            return self.__gauss
        elif activation == NodeActivation.SIGNUM:
            return self.__signum

    @staticmethod
    def __softmax(xs) -> List[float]:
        xs = [x - max(xs) for x in xs]
        xs_sum = sum(math.exp(x) for x in xs)
        return [math.exp(x) / xs_sum for x in xs]

    @staticmethod
    def __clamped(x: float) -> float:
        return -1 if x < -1 else (1 if x > 1 else x)

    @staticmethod
    def __inv(x: float) -> float:
        return -x

    @staticmethod
    def __abs(x: float) -> float:
        return abs(x)

    @staticmethod
    def __sin(x: float) -> float:
        return math.sin(math.pi * x)

    @staticmethod
    def __cos(x: float) -> float:
        return math.cos(math.pi * x)

    @staticmethod
    def __tanh(x: float) -> float:
        return -1 if x < -354 else 2 / (1 + math.e ** (-2*x)) - 1

    @staticmethod
    def __steepened_tanh(x: float) -> float:
        return -1 if x < -72 else 2 / (1 + math.e ** (-9.8*x)) - 1

    @staticmethod
    def __sigm(x: float) -> float:
        return 0 if x < -709 else 1 / (1 + math.e ** -x)

    @staticmethod
    def __steepened_sigm(x: float) -> float:
        return 0 if x < -144 else 1 / (1 + math.e ** (-4.9 * x))

    @staticmethod
    def __relu(x: float) -> float:
        return max(0, x)

    @staticmethod
    def __step(x: float) -> float:
        return 1.0 * (x > 0.0)

    @staticmethod
    def __lin(x: float) -> float:
        return x

    @staticmethod
    def __gauss(x: float) -> float:
        return math.e ** -(x ** 2)

    @staticmethod
    def __signum(x: float) -> float:
        return 0 if x == 0 else (-1 if x < 0 else 1)
