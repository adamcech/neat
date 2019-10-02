from gym_client.gym_client import GymClient


class PacmanClient(GymClient):

    def get_environment_name(self):
        return "MsPacman-ram-v0"

    def get_input_size(self) -> int:
        return 128

    def get_output_size(self) -> int:
        return 9

    def get_max_trials(self) -> int:
        return 1
