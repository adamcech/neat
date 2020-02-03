import random

import gym
import numpy as np
from typing import List, Tuple, Union, Any

from dataset.dataset import Dataset
from neat.ann.ann import Ann


class GymClient(Dataset):
    """Abstract class for gym_client clients
    """

    def __init__(self, environment_name: str, discrete: bool, inputs: int, outputs: int, bias: int, max_trials: int, max_episodes: int):

        self._environment_name = environment_name
        self._discrete = discrete

        self._inputs = inputs
        self._outputs = outputs
        self._bias = bias

        self._max_trials = max_trials
        self._max_episodes = max_episodes

        self._recalc_reward = self._recalc_reward_bipedal_walker_hardcore if environment_name == "BipedalWalkerHardcore-v2" else self._recalc_reward_default

        self.__bias_input = [1 for _ in range(self.get_bias_size())]  # type: List[float]

        if self.get_bias_size() == 0:
            if self.is_discrete():
                self._get_ann_action = self._get_ann_action_discrete_no_bias
            else:
                self._get_ann_action = self._get_ann_action_box_no_bias
        else:
            if self.is_discrete():
                self._get_ann_action = self._get_ann_action_discrete
            else:
                self._get_ann_action = self._get_ann_action_box

    def get_environment_name(self) -> str:
        return self._environment_name

    def is_discrete(self) -> bool:
        return self._discrete

    def get_input_size(self) -> int:
        return self._inputs

    def get_output_size(self) -> int:
        return self._outputs

    def get_bias_size(self) -> int:
        return self._bias

    def get_max_trials(self) -> int:
        return self._max_trials

    def get_max_episodes(self) -> int:
        return self._max_episodes

    def _get_ann_action_discrete(self, ann: Ann, observation: np.ndarray) -> Union[int, List[float]]:
        output = ann.calculate(np.append(observation, self.__bias_input))
        return output.index(max(output))

    def _get_ann_action_box(self, ann: Ann, observation: np.ndarray) -> Union[int, List[float]]:
        return ann.calculate(np.append(observation, self.__bias_input))

    @staticmethod
    def _get_ann_action_box_no_bias(ann: Ann, observation: np.ndarray) -> Union[int, List[float]]:
        return ann.calculate(observation)

    @staticmethod
    def _get_ann_action_discrete_no_bias(ann: Ann, observation: np.ndarray) -> Union[int, List[float]]:
        output = ann.calculate(observation)
        return output.index(max(output))

    @staticmethod
    def _recalc_reward_bipedal_walker_hardcore(x: Union[float, int]):
        return -10 if x == -100 else x

    @staticmethod
    def _recalc_reward_default(x: Union[float, int]):
        return x

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

    def get_fitness(self, ann: Ann, seed: Any = None) -> Tuple[float, Union[None, List[Any]], List[float], int]:
        scores = []
        envs = self.get_environment(seed)

        episodes = 0

        for env, s in envs:
            score = 0.0
            observation = env.reset()
            max_episodes = episodes + self.get_max_episodes()
            while episodes < max_episodes:
                episodes += 1
                observation, reward, done, info = env.step(self._get_ann_action(ann, observation))
                score += self._recalc_reward(reward)
                if done:
                    break
            env.close()
            scores.append(score)

        return sum(scores) / len(scores), [e[1] for e in envs], scores, episodes

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
                    score += self._recalc_reward(reward)
                env.close()
                print("Score: " + str(score) + "; " + str(s))
            counter += 1

    def get_random_seed(self, count: int) -> List[Any]:
        return random.sample(range(10000000), count)
