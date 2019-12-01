import gym

from gym_client.gym_client import GymClient


class CartPoleClient(GymClient):

    def get_environment(self) -> gym.Env:
        return gym.make("CartPole-v0")

    def get_max_trials(self) -> int:
        return 10  # 100

    def get_max_episodes(self) -> int:
        return 200

    def is_discrete(self) -> bool:
        return True

    def get_input_size(self) -> int:
        return 4

    def get_bias_size(self) -> int:
        return 0

    def get_output_size(self) -> int:
        return 2
