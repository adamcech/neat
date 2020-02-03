import os
from typing import Dict, Union, Any

from dataset.dataset_xor import DatasetXor
from dataset.gym_client import GymClient
from dataset.gym_client_creator import GymClientCreator
from neat.encoding.node_activation import NodeActivation


class Config:

    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.generation = 0
        self.evals = 0

        if type(config) != str and type(config) != dict:
            raise Exception("Config can be either Dict[str, Any] or string with path to config file.")

        if type(config) == str:
            config = self._config_file_to_dict(config)

        self._keys = {"dataset": (str, True, None, ["xor", "cartpole", "acrobot", "lunar_lander", "lunar_lander_continuous", "bipedal_walker", "bipedal_walker_hardcore", "pacman"]),
                      "activation": (str, True, "clamped", ["clamped", "tanh", "steepened_tanh", "sigm", "steepened_sigm"]),
                      "population_size": (int, True, None, None),

                      "seed_max": (int, False, None, None),
                      "seed_next": (int, False, None, None),
                      "seed_attempts": (int, False, None, None),
                      "seed_select": (int, False, None, None),

                      "crossover_f": (float, False, 0.6, None),

                      "compatibility_max_diffs": (int, False, 2, None),
                      "compatibility_threshold": (float, False, 0.95, None),
                      "compatibility_low": (float, False, -0.75, None),
                      "compatibility_low_crossover": (float, False, 0.5, None),
                      "compatibility_low_node_over_edge": (float, False, 0.1, None),

                      "max_generations": (int, False, 10000000000000, None),
                      "max_evals": (int, False, 10000000000000, None),
                      "max_score_termination": (float, False, None, None),

                      "max_trials": (int, False, None, None),
                      "max_episodes": (int, False, None, None),
                      "bias_nodes": (int, False, 0, None),

                      "species_max": (int, False, 4, None),
                      "species_elitism": (float, False, 0.2, None),
                      "species_mating": (float, False, 0.97, None),
                      "species_remove": (float, False, 0.0, None),
                      "species_representative_change": (int, False, 10, None),

                      "mutate_disable_lowest": (float, False, 0.0, None),
                      "mutate_add_node": (float, False, 0.05, None),
                      "mutate_add_edge": (float, False, 0.2, None),
                      "mutate_nde_perturbation_over_random": (float, False, 0.95, None),
                      "mutate_de_shift": (float, False, 0.3, None),
                      "mutate_de_random": (float, False, 0.01, None),
                      "mutate_enable": (float, False, 0.01, None),

                      "mutate_random_weight_mu": (float, False, 0.0, None),
                      "mutate_random_weight_sigm": (float, False, 1.0, None),
                      "mutate_perturbate_weight_mu": (float, False, 0.0, None),
                      "mutate_perturbate_weight_sigm": (float, False, 0.1, None),
                      "mutate_shift_weight_lower_bound": (float, False, 0.97, None),
                      "mutate_shift_weight_upper_bound": (float, False, 1.03, None),
                      "min_weight": (float, False, -8, None),
                      "max_weight": (float, False, 8, None),

                      "mp_max_proc": (int, False, 4, None),
                      "mp_step": (int, False, 5, None)}

        for key in config:
            if key not in self._keys:
                raise Exception("Invalid Key: " + str(key) + "\n" + self._get_key_info_message())
            if type(config[key]) != self._keys[key][0]:
                try:
                    config[key] = self._keys[key][0](config[key])
                except TypeError:
                    raise Exception("Invalid type " + str(type(config[key])) + " of Key: " + str(key) + "\n" + self._get_key_info_message())
            if self._keys[key][3] is not None:
                if config[key] not in self._keys[key][3]:
                    raise Exception("Invalid Value " + str(config[key]) + " of Key: " + str(key) + "\n" + self._get_key_info_message())

        for key in self._keys:
            if key not in config and self._keys[key][1]:
                raise Exception("Missing key: " + str(key) + "\n" + self._get_key_info_message())

        self.config = config

        # Not required params
        self.max_score_termination = self._get_with_default_value("max_score_termination")
        self.max_generations = self._get_with_default_value("max_generations")
        self.max_evals = self._get_with_default_value("max_evals")
        self.max_trials = self._get_with_default_value("max_trials")
        self.max_episodes = self._get_with_default_value("max_episodes")
        self.bias_nodes = self._get_with_default_value("bias_nodes")
        self.species_remove = self._get_with_default_value("species_remove")
        self.species_representative_change = self._get_with_default_value("species_representative_change")
        self.mutate_random_weight_mu = self._get_with_default_value("mutate_random_weight_mu")
        self.mutate_random_weight_sigm = self._get_with_default_value("mutate_random_weight_sigm")
        self.mutate_perturbate_weight_mu = self._get_with_default_value("mutate_perturbate_weight_mu")
        self.mutate_perturbate_weight_sigm = self._get_with_default_value("mutate_perturbate_weight_sigm")
        self.mutate_shift_weight_lower_bound = self._get_with_default_value("mutate_shift_weight_lower_bound")
        self.mutate_shift_weight_upper_bound = self._get_with_default_value("mutate_shift_weight_upper_bound")
        self.min_weight = self._get_with_default_value("min_weight")
        self.max_weight = self._get_with_default_value("max_weight")
        self.compatibility_low = self._get_with_default_value("compatibility_low")
        self.compatibility_low_crossover = self._get_with_default_value("compatibility_low_crossover")
        self.compatibility_low_node_over_edge = self._get_with_default_value("compatibility_low_node_over_edge")
        self.species_mating = self._get_with_default_value("species_mating")

        self.mutate_disable_lowest = self._get_with_default_value("mutate_disable_lowest")
        self.mutate_add_node = self._get_with_default_value("mutate_add_node")
        self.mutate_add_edge = self._get_with_default_value("mutate_add_edge")
        self.mutate_nde_perturbation_over_random = self._get_with_default_value("mutate_nde_perturbation_over_random")
        self.mutate_de_shift = self._get_with_default_value("mutate_de_shift")
        self.mutate_de_random = self._get_with_default_value("mutate_de_random")
        self.mutate_enable = self._get_with_default_value("mutate_enable")
        self.species_max = self._get_with_default_value("species_max")
        self.compatibility_max_diffs = self._get_with_default_value("compatibility_max_diffs")
        self.compatibility_threshold = self._get_with_default_value("compatibility_threshold")
        self.crossover_f = self._get_with_default_value("crossover_f")
        self.species_elitism = self._get_with_default_value("species_elitism")

        self.seed_max = self._get_with_default_value("seed_max")
        self.seed_next = self._get_with_default_value("seed_next")
        self.seed_attempts = self._get_with_default_value("seed_attempts")
        self.seed_select = self._get_with_default_value("seed_select")

        self.mp_max_proc = self._get_with_default_value("mp_max_proc")
        self.mp_step = self._get_with_default_value("mp_step")

        if self.max_weight < self.min_weight:
            raise Exception("Max weight must be bigger than min weight")

        self.population_size = self.config["population_size"]

        if self.config["activation"] == "clamped":
            self.activation = NodeActivation.CLAMPED
        elif self.config["activation"] == "tanh":
            self.activation = NodeActivation.TANH
        elif self.config["activation"] == "steepened_tanh":
            self.activation = NodeActivation.STEEPENED_TANH
        elif self.config["activation"] == "sigm":
            self.activation = NodeActivation.SIGM
        elif self.config["activation"] == "steepened_sigm":
            self.activation = NodeActivation.STEEPENED_SIGM
        else:
            raise Exception("Error parsing node activation\n" + self._get_key_info_message())

        self.dataset_name = self.config["dataset"]
        if self.config["dataset"] == "xor":
            self.dataset = DatasetXor(self.bias_nodes)
        elif self.config["dataset"] == "cartpole":
            self.dataset = GymClientCreator.create_cart_pole(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "acrobot":
            self.dataset = GymClientCreator.create_acrobot(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "lunar_lander":
            self.dataset = GymClientCreator.create_lunar_lander(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "lunar_lander_continuous":
            self.dataset = GymClientCreator.create_lunar_lander_continuous(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "bipedal_walker":
            self.dataset = GymClientCreator.create_bipedal_walker(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "bipedal_walker_hardcore":
            self.dataset = GymClientCreator.create_bipedal_walker_hardcore(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "pacman":
            self.dataset = GymClientCreator.create_pacman(self.bias_nodes, self.max_trials, self.max_episodes)
        else:
            raise Exception("Error parsing dataset\n" + self._get_key_info_message())

        self.input_nodes = self.dataset.get_input_size()
        self.bias_nodes = self.dataset.get_bias_size()
        self.output_nodes = self.dataset.get_output_size()

        if type(self.dataset) == GymClient:
            self.max_trials = self.dataset.get_max_trials()
            self.max_episodes = self.dataset.get_max_episodes()
            self.discrete = self.dataset.is_discrete()
            self.environment_name = self.dataset.get_environment_name()
        else:
            self.max_trials = None
            self.max_episodes = None
            self.discrete = None
            self.environment_name = None

        self.input_nodes_id = [i for i in range(self.input_nodes)]
        self.bias_nodes_id = [i for i in range(self.input_nodes, self.input_nodes + self.bias_nodes)]
        self.output_nodes_id = [i for i in range(self.input_nodes + self.bias_nodes, self.input_nodes + self.bias_nodes + self.output_nodes)]

        self.dir_path = os.getcwd() + os.path.sep + "files_"
        dir_counter = 0

        while True:
            if os.path.exists(self.dir_path + str(dir_counter)):
                dir_counter += 1
            else:
                self.dir_path += str(dir_counter)
                break

    def _get_with_default_value(self, key: str) -> Any:
        return self.config[key] if self.config.get(key) is not None else self._keys[key][2]

    def _get_key_info_message(self) -> str:
        msg = "Key: (Type, Required Key, Default Value, Possible Values)\n"
        for key in self._keys:
            msg += str(key) + ": " + str(self._keys[key]) + "\n"
        return msg

    def _config_file_to_dict(self, path: str) -> Union[Dict[str, Any], str]:
        conf = {}

        with open(path, "r") as f:
            content = f.read()
            lines = content.split("\n")

            for line in lines:
                if len(line) == 0 or line[0] == "#":
                    continue

                key_and_val = self._replace_spaces(line).split(":")

                if len(key_and_val) != 2:
                    continue

                key = key_and_val[0]
                val = key_and_val[1]

                if key in conf:
                    raise Exception("Multiple occurence of param " + str(key))

                conf[key] = val

        return conf

    def __str__(self):
        msg = ""
        for key in self.config:
            msg += str(key) + ": " + str(self.config[key]) + "\n"
        return msg

    def _replace_spaces(self, s: str) -> str:
        return self._replace_spaces(s.replace(" ", "")) if ' ' in s else s
