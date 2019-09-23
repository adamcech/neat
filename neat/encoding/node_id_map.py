from typing import List

from neat.encoding.node import Node


class NodeIdMap:
    """Node id map utility
    """

    def __init__(self):
        self._node_id_counter = -1
        self._keys = {}

    def get_id(self, input_id: int, output_id: int, existing_nodes: List[Node]) -> int:
        key = NodeIdMap.__get_key(input_id, output_id)
        id_list = self._keys.get(key)

        if id_list is None:
            new_id = self.next_id()
            self._keys[key] = [new_id]
            return new_id
        else:
            existing_id = [e.id for e in existing_nodes]

            if all(node_id in existing_id for node_id in id_list):
                new_id = self.next_id()
                id_list.append(new_id)
                return new_id
            else:
                for node_id in id_list:
                    if node_id not in existing_id:
                        return node_id

    def next_id(self) -> int:
        self._node_id_counter += 1
        return self._node_id_counter

    @staticmethod
    def __get_key(input_id: int, output_id: int) -> str:
        return str(min(input_id, output_id)) + "-" + str(max(input_id, output_id))
