import gym

from gym_client.gym_client import GymClient


class LunarLanderClient(GymClient):

    def get_environment(self) -> gym.Env:
        return gym.make("LunarLander-v2")

    def is_discrete(self) -> bool:
        return True

    def get_max_trials(self) -> int:
        return 3

    def get_max_episodes(self) -> int:
        return 500

    def get_input_size(self) -> int:
        return 8

    def get_bias_size(self) -> int:
        return 1

    def get_output_size(self) -> int:
        return 4
