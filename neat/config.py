from typing import Dict, Union, Any

from dataset.dataset_iris import DatasetIris
from dataset.dataset_xor import DatasetXor
from dataset.gym_client import GymClient
from dataset.gym_client_creator import GymClientCreator
from neat.ann.target_function import TargetFunction
from neat.encoding.node_activation import NodeActivation


class Config:

    def __init__(self, config: Union[str, Dict[str, Any]]):
        self.generation = 0
        self.evals = 0

        if type(config) != str and type(config) != dict:
            raise Exception("Config can be either Dict[str, Any] or string with path to config file.")

        if type(config) == str:
            config = self._config_file_to_dict(config)

        # Type, Mandatory, Default Value, Possible Values
        self._keys = {"dataset": (str, True, None, ["xor", "flappy_bird", "swingup", "bipedal_walker_both", "iris", "car_racing", "vae_racing", "cartpole", "double_cartpole", "acrobot", "lunar_lander", "lunar_lander_continuous", "snake", "bipedal_walker", "bipedal_walker_hardcore", "pacman", "inverted_double_pendulum", "mountain_car", "mountain_car_continuous"]),
                      "activation": (str, True, "tanh", ["clamped", "tanh", "steepened_tanh", "sigm", "steepened_sigm", "relu"]),
                      "target_function": (str, True, "tanh", ["lin", "steepened_tanh", "tanh", "sigm", "steepened_sigm", "clamped", "softmax"]),
                      "population_size": (int, True, None, None),
                      "start_min": (str, False, True, ["True", "False"]),
                      "max_layers": (int, False, 3, None),

                      "seed_max": (int, False, 1, None),
                      "seed_test": (int, False, 100, None),
                      "seed_next": (int, False, 100, None),
                      "seed_attempts": (int, False, 100, None),
                      "seed_select": (int, False, 1, None),
                      "seed_select_random": (int, False, 1, None),

                      "stagnation_ind": (int, 10, None, None),

                      "learning_period": (int, False, 50, None),

                      "compatibility_max_diffs": (int, False, 4, None),
                      "compatibility_species_max_diffs": (int, False, 10, None),

                      "max_generations": (int, False, 10000000000000, None),
                      "max_evals": (int, False, 10000000000000, None),
                      "max_score_termination": (float, False, None, None),

                      "max_trials": (int, False, None, None),
                      "max_episodes": (int, False, None, None),
                      "bias_nodes": (int, False, 0, None),

                      "species_max": (int, False, 4, None),
                      "species_elitism": (float, False, 0.2, None),

                      "archivate_prop": (float, False, 0.03, None),
                      "mutate_add_node": (float, False, 0.05, None),
                      "mutate_add_edge": (float, False, 0.2, None),
                      "mutate_activation_change": (float, False, 0.05, None),
                      "mutate_nde_perturbation_over_random": (float, False, 0.95, None),
                      "mutate_de_shift": (float, False, 0.05, None),
                      "mutate_de_random": (float, False, 0.03, None),
                      "mutate_de_perturbate": (float, False, 0.03, None),

                      "mutate_shift_weight_lower_bound": (float, False, 0.8, None),
                      "mutate_shift_weight_upper_bound": (float, False, 1.0, None),
                      "weight_scale": (float, False, 10.0, None),
                      "weight_random_scale": (str, False, "1.0", None),
                      "weight_pert_scale": (str, False, "0.5", None),

                      "mp_max_proc": (int, False, 4, None),
                      "mp_step": (int, False, 5, None),

                      "cluster_evaluation": (str, False, False, ["True", "False"]),
                      "ray_info_loc": (str, False, None, None),
                      "cluster_nodes_loc": (str, False, None, None),
                      "cluster_main_loc": (str, False, None, None),
                      "cluster_main_max_load": (int, False, 10, None),

                      "test_size": (int, False, 10, None),
                      "test_attempts": (int, False, 100, None),
                      "test_counter": (int, False, 10, None),

                      "work_dir": (str, False, None, None),
                      "walltime_sec": (int, False, None, None)}

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
            if key not in config and type(self._keys[key][1]) == bool and self._keys[key][1]:
                raise Exception("Missing key: " + str(key) + "\n" + self._get_key_info_message())

        self.config = config
        self.learning_period = self._get_with_default_value("learning_period")
        self.stagnation_ind = self._get_with_default_value("stagnation_ind")

        self.max_score_termination = self._get_with_default_value("max_score_termination")
        self.max_generations = self._get_with_default_value("max_generations")
        self.max_evals = self._get_with_default_value("max_evals")
        self.max_trials = self._get_with_default_value("max_trials")
        self.max_episodes = self._get_with_default_value("max_episodes")

        self.bias_nodes = self._get_with_default_value("bias_nodes")
        self.start_min = self._get_with_default_value("start_min")
        self.start_min = True if self.start_min == "True" else False
        self.max_layers = self._get_with_default_value("max_layers")

        self.archivate_prop = self._get_with_default_value("archivate_prop")
        self.mutate_add_node = self._get_with_default_value("mutate_add_node")
        self.mutate_add_edge = self._get_with_default_value("mutate_add_edge")
        self.mutate_activation_change = self._get_with_default_value("mutate_activation_change")
        self.mutate_de_perturbate = self._get_with_default_value("mutate_de_perturbate")
        self.mutate_de_shift = self._get_with_default_value("mutate_de_shift")
        self.mutate_de_random = self._get_with_default_value("mutate_de_random")
        self.mutate_nde_perturbation_over_random = self._get_with_default_value("mutate_nde_perturbation_over_random")

        self.mutate_shift_weight_lower_bound = self._get_with_default_value("mutate_shift_weight_lower_bound")
        self.mutate_shift_weight_upper_bound = self._get_with_default_value("mutate_shift_weight_upper_bound")
        self.weight_scale = abs(self._get_with_default_value("weight_scale"))
        self.min_weight = -self.weight_scale
        self.max_weight = self.weight_scale
        self.weight_random_scale = [float(x) for x in str(self._get_with_default_value("weight_random_scale")).split(",")]
        self.weight_pert_scale = [float(x) for x in str(self._get_with_default_value("weight_pert_scale")).split(",")]

        self.species_max = self._get_with_default_value("species_max")
        self.species_elitism = self._get_with_default_value("species_elitism")
        self.compatibility_max_diffs = self._get_with_default_value("compatibility_max_diffs")
        self.compatibility_species_max_diffs = self._get_with_default_value("compatibility_species_max_diffs")

        self.seed_max = self._get_with_default_value("seed_max")
        self.seed_test = self._get_with_default_value("seed_test")
        self.seed_next = self._get_with_default_value("seed_next")
        self.seed_attempts = self._get_with_default_value("seed_attempts")
        self.seed_select = self._get_with_default_value("seed_select")
        self.seed_select_random = self._get_with_default_value("seed_select_random")

        self.mp_max_proc = self._get_with_default_value("mp_max_proc")
        self.mp_step = self._get_with_default_value("mp_step")

        self.cluster_evaluation = self._get_with_default_value("cluster_evaluation")
        self.cluster_evaluation = True if self.cluster_evaluation == "True" else False

        self.ray_info_loc = self._get_with_default_value("ray_info_loc")
        self.cluster_nodes_loc = self._get_with_default_value("cluster_nodes_loc")
        self.cluster_main_loc = self._get_with_default_value("cluster_main_loc")
        self.cluster_main_max_load = self._get_with_default_value("cluster_main_max_load")

        self.jade_c = 0.1

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
        elif self.config["activation"] == "relu":
            self.activation = NodeActivation.RELU
        else:
            raise Exception("Error parsing node activation\n" + self._get_key_info_message())

        if self.config["target_function"] == "clamped":
            self.target_function = TargetFunction.CLAMPED
        elif self.config["target_function"] == "relu":
            self.target_function = TargetFunction.RELU
        elif self.config["target_function"] == "steepened_tanh":
            self.target_function = TargetFunction.STEEPENED_TANH
        elif self.config["target_function"] == "tanh":
            self.target_function = TargetFunction.TANH
        elif self.config["target_function"] == "steepened_sigm":
            self.target_function = TargetFunction.STEEPENED_SIGM
        elif self.config["target_function"] == "sigm":
            self.target_function = TargetFunction.SIGM
        elif self.config["target_function"] == "softmax":
            self.target_function = TargetFunction.SOFTMAX
        elif self.config["target_function"] == "lin":
            self.target_function = TargetFunction.LIN

        self.dataset_name = self.config["dataset"]
        if self.config["dataset"] == "xor":
            self.dataset = DatasetXor(self.bias_nodes)
            self.cluster_evaluation = False
        elif self.config["dataset"] == "iris":
            self.dataset = DatasetIris(self.bias_nodes)
            self.cluster_evaluation = False
        elif self.config["dataset"] == "vae_racing":
            self.dataset = GymClientCreator.create_vae_racing(self.bias_nodes)
        elif self.config["dataset"] == "flappy_bird":
            self.dataset = GymClientCreator.create_flappy_bird(self.bias_nodes)
        elif self.config["dataset"] == "car_racing":
            self.dataset = GymClientCreator.create_car_racing(self.bias_nodes)
        elif self.config["dataset"] == "bipedal_walker_both":
            self.dataset = GymClientCreator.create_bipedal_walker_both(self.bias_nodes)
        elif self.config["dataset"] == "cartpole":
            self.dataset = GymClientCreator.create_cart_pole(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "swingup":
            self.dataset = GymClientCreator.create_cart_pole_swingup(self.bias_nodes)
        elif self.config["dataset"] == "double_cartpole":
            self.dataset = GymClientCreator.create_double_cart_pole(self.bias_nodes, self.max_trials, self.max_episodes)
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
        elif self.config["dataset"] == "inverted_double_pendulum":
            self.dataset = GymClientCreator.create_double_inverted_pendulum(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "mountain_car_continuous":
            self.dataset = GymClientCreator.create_mountain_car_continuous(self.bias_nodes, self.max_trials, self.max_episodes)
        elif self.config["dataset"] == "mountain_car":
            self.dataset = GymClientCreator.create_mountain_car_continuous(self.bias_nodes, self.max_trials, self.max_episodes)
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

        self.walltime_sec = self._get_with_default_value("walltime_sec")
        self.work_dir = self._get_with_default_value("work_dir")

        self.tcp_port = 49152
        self.done = False
        self.done_genotype = None

    def next_tcp_port(self):
        if self.tcp_port + 1 > 65535:
            self.tcp_port = 49152

        self.tcp_port += 1

    def is_seed_params_set(self):
        return self.seed_next is not None and self.seed_max is not None and self.seed_attempts is not None and self.seed_select is not None

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
