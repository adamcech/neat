from gym_client.gym_client import GymClient


class PoleBalancingClient(GymClient):

    def get_environment_name(self):
        return "CartPole-v0"

    def get_input_size(self) -> int:
        return 4

    def get_output_size(self) -> int:
        return 2

    def get_max_trials(self) -> int:
        return 100
