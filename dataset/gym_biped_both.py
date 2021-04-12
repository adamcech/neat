from typing import Union, Type, List, Any, Tuple

from dataset.dataset import Dataset
from dataset.gym_client import GymClient
from neat.ann.ann import Ann


class GymBipedBoth(Dataset):

    def __init__(self, environment_name: str, bias: int):
        self._environment_name = environment_name
        self._bias = bias

        self.envs = [
            GymClient("BipedalWalkerHardcore-v3", False, 24, 4, bias, 1, 2000),
            GymClient("BipedalWalker-v3", False, 24, 4, bias, 1, 1600)
        ]

    def get_input_size(self) -> int:
        return 24

    def get_output_size(self) -> int:
        return 4

    def get_bias_size(self) -> int:
        return self._bias

    def get_fitness(self, ann: Ann, seed: Any = None, **kwargs) -> Tuple[float, Union[None, List[Any]], List[float], int]:
        score, scores, episodes = 0, [0 for _ in seed], 0

        for i in range(len(self.envs)):
            s, env, ss, ep = self.envs[i].get_fitness(ann, seed)
            episodes += ep
            scores = [scores[j] + ss[j] for j in range(len(seed))]

        scores = [s / (len(self.envs) * len(seed)) for s in scores]
        return sum(scores) / len(scores), seed, scores, episodes

    def render(self, ann: Ann, seed: Any = None, **kwargs) -> None:
        for s in seed:
            for env in self.envs:
                env.render(ann, [s], loops=1)

    def get_random_seed(self, count: int) -> Union[List[Any], None]:
        return self.envs[0].get_random_seed(count)

    def get_seed_type(self) -> Union[None, Type[list]]:
        return self.envs[0].get_seed_type()
