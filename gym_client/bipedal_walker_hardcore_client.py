import gym

from gym_client.gym_client import GymClient


class BipedalWalkerHardcoreClient(GymClient):

    def get_environment(self) -> gym.Env:
        return gym.make("BipedalWalkerHardcore-v2")

    def is_discrete(self) -> bool:
        return False

    def get_max_trials(self) -> int:
        return 1

    def get_max_episodes(self) -> int:
        return 2000

    def get_input_size(self) -> int:
        return 24

    def get_bias_size(self) -> int:
        return 2

    def get_output_size(self) -> int:
        return 4
