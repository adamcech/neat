import pickle
import socket
from enum import Enum
from typing import List, Tuple, Union, Any

from evaluators.cluster.my_socket_functions import MySocketFunctions
import multiprocessing as mp

from neat.encoding.genotype import Genotype


class SocketState(Enum):
    READY = 1
    WORKING = 2
    DONE = 3


class ClusterNodeClient:

    def __init__(self, max_load: int, hostname: str, port: int):
        self.hostname = hostname
        self.port = port

        self._state = SocketState.READY
        self.results = []

        self.request_q = mp.Queue()
        self.request_p = mp.Process()

        self.max_load = max_load

    def request(self, request: List[Tuple[int, Genotype, Union[List[Any], None]]]) -> None:
        self._state = SocketState.WORKING

        self.request_q = mp.Queue()

        self.request_p = mp.Process(target=self.__handle, args=(self.hostname, self.port, self.request_q, request))
        self.request_p.start()

    @staticmethod
    def __handle(hostname: str, port: int, q: mp.Queue, request: List[Tuple[int, Genotype, Union[List[Any], None]]]):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((hostname, port))

        MySocketFunctions.send(s, pickle.dumps(request))
        q.put(pickle.loads(MySocketFunctions.recv(s)))

        s.close()

    def _check_state(self):
        if not self.request_p.is_alive():
            self.request_p.join()

            self.results = self.request_q.get()
            self.request_q.close()
            self._state = SocketState.DONE

    def get_results(self) -> List[Tuple[int, float, Union[None, List[Any]], int]]:
        results = self.results
        self.results = []
        self._state = SocketState.READY
        return results

    def is_ready(self):
        return self._state == SocketState.READY

    def is_working(self):
        return self._state == SocketState.WORKING

    def is_done(self):
        if self.is_working():
            self._check_state()
        return self._state == SocketState.DONE

    def __str__(self):
        return "Cluster Node = " + self.hostname + ":" + str(self.port)

    def __repr__(self):
        return "Cluster Node = " + self.hostname + ":" + str(self.port)
