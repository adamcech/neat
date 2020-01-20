class InnovationMap:
    """Utility for keeping track of innovation numbers
    """

    def __init__(self):
        self._innovation_counter = -1
        self._node_keys = {}
        self._connection_keys = {}

    def get_node_innovation(self, input_id: int, output_id: int, existing_nodes: list) -> int:
        key = InnovationMap.__get_key(input_id, output_id)
        id_list = self._node_keys.get(key)

        if id_list is None:
            new_id = self.next_innovation()
            self._node_keys[key] = [new_id]
            return new_id
        else:
            existing_id = [e.id for e in existing_nodes]

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
