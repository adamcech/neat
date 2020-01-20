import gym
import numpy as np
from typing import List, Tuple, Union, Any

from dataset.dataset import Dataset
from neat.ann.ann import Ann


class GymClientText(Dataset):
    """Abstract class for gym_client clients
    """

    def __init__(self):
        self.__bias_input = [1 for _ in range(self.get_bias_size())]  # type: List[float]

    def get_environment(self, seed: List[Any] = None) -> List[Tuple[gym.Env, Any]]:
        env_name = self.get_environment_name()
        envs = []

        if type(seed) == list:
            for s in seed:
                env = gym.make(env_name)
                env_seed = env.seed(s)[0]
                envs.append((env, env_seed))
        else:
            for i in range(self.get_max_trials()):
                env = gym.make(env_name)
                env_seed = env.seed()[0]
                envs.append((env, env_seed))

        return envs

    def get_environment_name(self) -> str:
        raise NotImplementedError()

    def get_max_trials(self) -> int:
        raise NotImplementedError()

    def get_max_episodes(self) -> int:
        raise NotImplementedError()

    def is_discrete(self) -> bool:
        raise NotImplementedError()

    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_bias_size(self) -> int:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_output_groups(self) -> Tuple[int, int, int]:
        raise NotImplementedError()

    def recalc_reawrd(self, reward: Union[int, float]) -> Union[int, float]:
        return reward

    def _get_ann_action(self, ann: Ann, observation: np.ndarray) -> Tuple[int, int, int]:
        output = ann.calculate(np.append(observation, self.__bias_input))

        output_groups = self.get_output_groups()
        group0 = output[0:output_groups[0]]
        group1 = output[output_groups[0]:output_groups[0] + output_groups[1]]
        group2 = output[output_groups[0] + output_groups[1]:sum(output_groups)]

        return group0.index(max(group0)), group1.index(max(group1)), group2.index(max(group2))

    def get_fitness(self, ann: Ann, seed: Any = None) -> Tuple[float, Union[None, List[Any]], List[float]]:
        scores = []
        envs = self.get_environment(seed)

        for env, s in envs:
            score = 0.0
            observation = env.reset()
            for _ in range(self.get_max_episodes()):
                observation, reward, done, info = env.step(self._get_ann_action(ann, observation))
                score += self.recalc_reawrd(reward)
                if done:
                    break
            env.close()
            scores.append(score)

        return sum(scores) / len(scores), [e[1] for e in envs], scores

    def render(self, ann: Ann, seed: Any = None, **kwargs):
        loops = kwargs.get("loops", None)
        counter = 0

        while True if loops is None else counter < loops:
            envs = self.get_environment(seed)
            for env, s in envs:
                observation, done, score = env.reset(), False, 0
                while not done:
                    env.render()
                    action = self._get_ann_action(ann, observation)
                    observation, reward, done, info = env.step(action)
                    score += self.recalc_reawrd(reward)
                env.close()
                print("Score: " + str(score) + "; " + str(s))
            counter += 1

    def get_env_info(self):
        env = self.get_environment()
        print("Observation (input):   " + str(env.observation_space))
        print("Actions (outputs):     " + str(env.action_space))
