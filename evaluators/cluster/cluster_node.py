import pickle
import socket

from evaluators.cluster.my_socket_functions import MySocketFunctions
from evaluators.results_calculator import ResultsCalculator

import multiprocessing as mp


class ClusterNode:

    def __init__(self, config, run_parallel: bool = False):
        self.config = config
        self.dir_path_redirect = config.work_dir
        self.results_calculator = ResultsCalculator(self.config)

        mp.Process(target=ClusterNode.run, args=(self,)).start() if run_parallel else self.run()

    def run(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((socket.gethostname(), self.config.tcp_port))
        s.listen()

        print("NODE STARTED:   " + socket.gethostname() + ":" + str(self.config.tcp_port))
        while True:
            client, address = s.accept()
            MySocketFunctions.send(client, pickle.dumps(self.results_calculator.calculate(pickle.loads(MySocketFunctions.recv(client)))))
            client.close()
