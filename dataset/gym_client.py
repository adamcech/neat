import math
import random
import time

import gym
import gym_cartpole_swingup

from gym import wrappers

import numpy as np
from typing import List, Tuple, Union, Any, Type

from dataset.dataset import Dataset
from dataset.double_cartpole import DoubleCartPole
from neat.ann.ann import Ann

# Virtual display
# from pyvirtualdisplay import Display
#
# virtual_display = Display(visible=0)
# virtual_display.start()

gym.logger.set_level(40)


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

        if environment_name == "BipedalWalkerHardcore-v3":
            self._recalc_reward = self._recalc_reward_bipedal_walker_hardcore
            self._recalc_reward_end = self._recalc_reward_end_bipedal_walker_hardcore
        elif environment_name == "BipedalWalker-v3":
            self._recalc_reward = self._recalc_reward_bipedal_walker_hardcore
            self._recalc_reward_end = self._recalc_reward_end_bipedal_walker
        else:
            self._recalc_reward = self._recalc_reward_default
            self._recalc_reward_end = self._recalc_reward_end_default

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
    def _recalc_reward_bipedal_walker_hardcore(score, reward, action):
        return score + reward
        if reward == -100 and score > 0:
            return score - max(10, score * 0.25)
        else:
            # return score + (reward * 0.5) if reward > 0 and any(a >= 1 or a <= -1 for a in action) else score + reward
            return score + reward

    @staticmethod
    def _recalc_reward_bipedal_walker(score, reward, action):
        p = 0.5 / len(action)
        penalty = 1.0 - sum(p for a in action if a == 0 or a >= 1 or a <= -1)
        return score + (reward * penalty) if reward > 0 else score + reward

    @staticmethod
    def _recalc_reward_default(score, reward, penalty=1):
        return score + (reward * penalty)

    @staticmethod
    def _recalc_reward_end_default(score):
        return score

    @staticmethod
    def _recalc_reward_end_bipedal_walker_hardcore(score):
        return score if score > 0 else 0.00001

    @staticmethod
    def _recalc_reward_end_bipedal_walker(score):
        return score
        return score if score > -10 else -10

    def get_environment(self, seed: List[Any] = None) -> List[Tuple[gym.Env, Any]]:
        env_name = self.get_environment_name()
        envs = []

        if type(seed) == list:
            for s in seed:
                env = self._make(env_name)
                env_seed = env.seed(s)[0]
                envs.append((env, env_seed))
        else:
            for i in range(self.get_max_trials()):
                env = self._make(env_name)
                env_seed = env.seed()[0]
                envs.append((env, env_seed))

        return envs

    @staticmethod
    def _make(env_name: str):
        if env_name == "DoubleCartPole":
            env = DoubleCartPole()
        else:
            env = gym.make(env_name)

        return env

    def test_env(self, steps: int = None):
        env = self._make(self._environment_name)
        env.reset()

        # print("Action Space: ", env.action_space)
        # print("Observation Space: ", env.observation_space)

        score = 0
        s = 0

        steps = self._max_episodes if steps is None else steps
        for step in range(steps):
            s += 1
            env.render()
            obs, reward, done, _ = env.step(env.action_space.sample())  # take a random action
            print(step, reward)
            print(obs)

            score += reward
            if done:
                print("TEST DONE ##### Reward: ", score, "Steps", step)
                break

        env.close()
        return score, s

    def get_fitness(self, ann: Ann, seed: Any = None, **kwargs) -> Tuple[float, Union[None, List[Any]], List[float], int]:
        allow_penalty = kwargs.get("allow_penalty", True)

        scores = []
        envs = self.get_environment(seed)

        episodes = 0

        for env, s in envs:
            score = 0.0
            observation = env.reset()
            max_episodes = episodes + self.get_max_episodes()
            while episodes < max_episodes:
                episodes += 1
                action = self._get_ann_action(ann, observation)

                observation, reward, done, info = env.step(action)
                score = self._recalc_reward(score, reward, action) if allow_penalty else score + reward

                if done:
                    if allow_penalty:
                        score = self._recalc_reward_end(score)
                    break
            env.close()
            scores.append(score)

        return sum(scores) / len(scores), [e[1] for e in envs], scores, episodes

    def render(self, ann: Ann, seed: Any = None, **kwargs):
        loops = kwargs.get("loops", None)

        envs = self.get_environment(seed)
        for env, s in envs:
            episodes = 0
            observation, done, score, score_base = env.reset(), False, 0, 0
            while not done:
                episodes += 1
                env.render()
                action = self._get_ann_action(ann, observation)
                for a in action:
                    string = str(round(a, 1)) + " " + ("T" if (a >= 1 or a <= -1) else "F") + " "
                    # print(string.rjust(5), end="   ")

                observation, reward, done, info = env.step(action)
                score = self._recalc_reward(score, reward, action)

                # print("     ", round(score_base, 1), "     ", str(round(score, 1)).ljust(10), round(reward, 1))

                score_base += reward

            env.close()
            score = self._recalc_reward_end(score)
            print("Score = " + str(round(score, 2)).ljust(6) + "; Score_base = " + str(score_base) + "; SEED = " + str(s).ljust(7) + "; Episodes = " + str(episodes).ljust(6) + "; ADJ = " + str(round((score ** 2) / episodes * (1 if score > 0 else -1), 4)))

    def get_seed_type(self) -> Union[None, Type[list]]:
        return list

    def get_random_seed(self, count: int) -> List[Any]:
        return random.sample(range(10000000), count)
