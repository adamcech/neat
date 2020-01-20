from typing import Union, Tuple

from gym_client.gym_client import GymClient
from gym_client.gym_client_text import GymClientText


class BipedalWalkerClient(GymClientText):

    def get_environment_name(self) -> str:
        return "Copy-v0"

    def is_discrete(self) -> bool:
        return True

    def get_max_trials(self) -> int:
        return 1

    def get_max_episodes(self) -> int:
        return 200

    def get_input_size(self) -> int:
        return 6

    def get_bias_size(self) -> int:
        return 1

    def get_output_size(self) -> int:
        return 9

    def get_output_groups(self) -> Tuple[int, int, int]:
        return 2, 2, 5

    def recalc_reawrd(self, reward: Union[int, float]) -> Union[int, float]:
        return reward
