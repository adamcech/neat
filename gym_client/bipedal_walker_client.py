import gym

from gym_client.gym_client import GymClient


class BipedalWalkerClient(GymClient):

    def get_environment(self) -> gym.Env:
        return gym.make("BipedalWalker-v2")

    def is_discrete(self) -> bool:
        return False

    def get_max_trials(self) -> int:
        return 1 if self._max_trials is None else self._max_trials

    def get_max_episodes(self) -> int:
        return 1600

    def get_input_size(self) -> int:
        return 24

    def get_bias_size(self) -> int:
        return 0

    def get_output_size(self) -> int:
        return 4
