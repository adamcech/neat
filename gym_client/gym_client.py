import gym
import numpy as np
from typing import List, Tuple

from dataset.dataset import Dataset
from neat.ann.ann import Ann


class GymClient(Dataset):
    """Abstract class for gym_client clients
    """
    def __init__(self):
        self.__bias_input = [1 for _ in range(self.get_bias_size())]  # type: List[float]
        self._max_trials = None

    def get_environment(self) -> gym.Env:
        raise NotImplementedError()

    def get_max_trials(self) -> int:
        raise NotImplementedError()

    def set_max_trials(self, max_trials: int) -> None:
        self._max_trials = max_trials

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

    def _get_ann_action(self, ann: Ann, observation: np.ndarray) -> int:
        output = ann.calculate(np.append(observation, self.__bias_input))
        return output.index(max(output)) if self.is_discrete() else output

    def get_fitness(self, ann: Ann) -> float:
        score = 0.0
        env = self.get_environment()

        for trials in range(self.get_max_trials()):
            observation = env.reset()
            for _ in range(self.get_max_episodes()):
                observation, reward, done, info = env.step(self._get_ann_action(ann, observation))
                score += reward
                if done:
                    break

        env.close()
        return score / self.get_max_trials()

    def render(self, ann: Ann, **kwargs):
        env = self.get_environment()

        loops = kwargs.get("loops", None)
        counter = 0

        while True if loops is None else counter < loops:
            observation, done, score = env.reset(), False, 0
            while not done:
                env.render()
                observation, reward, done, info = env.step(self._get_ann_action(ann, observation))
                score += reward

            print("Score: " + str(score))

            counter += 1

        env.close()

    def get_env_info(self):
        env = self.get_environment()
        print("Observation (input):   " + str(env.observation_space))
        print("Actions (outputs):     " + str(env.action_space))
