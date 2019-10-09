from gym_client.gym_client import GymClient


class LunarLanderClient(GymClient):

    def get_environment_name(self) -> str:
        return "LunarLander-v2"

    def is_discrete(self) -> bool:
        return True

    def get_inputs(self) -> int:
        return 8

    def bias_nodes(self) -> int:
        return 1

    def get_output_size(self) -> int:
        return 4

    def get_max_trials(self) -> int:
        return 2

    def get_max_episodes(self) -> int:
        return 500
