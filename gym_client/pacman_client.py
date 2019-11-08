import gym

from gym_client.gym_client import GymClient


class PacmanClient(GymClient):

    def get_environment(self) -> gym.Env:
        return gym.make("MsPacman-ram-v0")

    def is_discrete(self) -> bool:
        return True

    def get_max_trials(self) -> int:
        return 2

    def get_max_episodes(self) -> int:
        return 10000

    def get_input_size(self) -> int:
        return 128

    def get_bias_size(self) -> int:
        return 2

    def get_output_size(self) -> int:
        return 9
