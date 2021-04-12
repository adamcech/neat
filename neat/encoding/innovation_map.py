from typing import Tuple, Union, Dict, List

from neat.encoding.node import Node


class InnovationMap:
    """Utility for keeping track of innovation numbers
    """

    def __init__(self):
        self._innovation_counter = -1
        self._node_keys = {}  # type: Dict[str, List[int]]
        self._connection_keys = {}  # type: Dict[str, int]
        self._node_direction = {}  # type: Dict[int, Tuple[int, int]]
        self._node_extends = {}  # type: Dict[int, List[int]]

    def next_extend_innovation(self, node_to_extend_id: int, existing_nodes: List[Node]) -> int:
        id_list = self._node_extends.get(node_to_extend_id)

        if id_list is None:
            new_id = self.next_innovation()
            self._node_extends[node_to_extend_id] = [new_id]
            return new_id
        else:
            existing_id = {e.id for e in existing_nodes}

            if all(node_id in existing_id for node_id in id_list):
                new_id = self.next_innovation()
                id_list.append(new_id)
                return new_id
            else:
                for node_id in id_list:
                    if node_id not in existing_id:
                        return node_id

    def get_node_info(self, node_id) -> Union[Tuple[int, int], None]:
        for key in self._node_keys:
            key_ids = self._node_keys[key]
            if node_id in key_ids:
                splited_key = str(key).split("-")
                node_input = int(splited_key[0])
                node_output = int(splited_key[1])
                return node_input, node_output
        return None

    def get_edge_info(self, innovation: int) -> Union[Tuple[int, int], None]:
        for key in self._connection_keys:
            key_innovation = self._connection_keys[key]
            if innovation == key_innovation:
                splited_key = str(key).split("-")
                node_input = int(splited_key[0])
                node_output = int(splited_key[1])
                return node_input, node_output
        return None

    def get_node_innovation(self, input_id: int, output_id: int, existing_nodes: List[Node]) -> int:
        new_id = self._get_node_innovation_impl(input_id, output_id, existing_nodes)

        if self._node_direction.get(new_id) is None:
            self._node_direction[new_id] = (input_id, output_id)

        return new_id

    def get_node_direction(self, node_id: int) -> Tuple[int, int]:
        return self._node_direction[node_id]

    def _get_node_innovation_impl(self, input_id: int, output_id: int, existing_nodes: List[Node]) -> int:
        key = InnovationMap.__get_key(input_id, output_id)
        id_list = self._node_keys.get(key)

        if id_list is None:
            new_id = self.next_innovation()
            self._node_keys[key] = [new_id]
            return new_id
        else:
            existing_id = {e.id for e in existing_nodes}

            if all(node_id in existing_id for node_id in id_list):
                new_id = self.next_innovation()
                id_list.append(new_id)
                return new_id
            else:
                for node_id in id_list:
                    if node_id not in existing_id:
                        return node_id

    def get_edge_innovation(self, input_id: int, output_id: int):
        key = InnovationMap.__get_key(input_id, output_id)
        innovation_number = self._connection_keys.get(key)

        if innovation_number is None:
            innovation_number = self.next_innovation()
            self._connection_keys[key] = innovation_number

        return innovation_number

    def next_innovation(self) -> int:
        self._innovation_counter += 1
        return self._innovation_counter

    @staticmethod
    def __get_key(input_id: int, output_id: int) -> str:
        return str(min(input_id, output_id)) + "-" + str(max(input_id, output_id))
