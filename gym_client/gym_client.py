import gym
import numpy

from dataset.dataset import Dataset
from neat.ann.ann import Ann


class GymClient(Dataset):
    """Abstract class for gym_client clients
    """
    def get_environment_name(self) -> int:
        raise NotImplementedError()

    def get_input_size(self) -> int:
        raise NotImplementedError()

    def get_output_size(self) -> int:
        raise NotImplementedError()

    def get_max_trials(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def get_ann_action(ann: Ann, observation: numpy.ndarray) -> int:
        output = ann.calculate(observation)
        return output.index(max(output))

    def get_fitness(self, ann: Ann) -> float:
        fitness = 0.0

        env = gym.make(self.get_environment_name())

        for trials in range(self.get_max_trials()):
            observation = env.reset()
            done = False
            while not done:
                observation, reward, done, info = env.step(self.get_ann_action(ann, observation))
                fitness += reward

        env.close()

        return fitness / self.get_max_trials()

    def render(self, ann: Ann, **kwargs):
        env = gym.make(self.get_environment_name())

        loops = kwargs.get("loops", None)
        counter = 0

        while self.is_rendering(loops, counter):
            observation = env.reset()
            done = False
            while not done:
                env.render()
                observation, reward, done, info = env.step(self.get_ann_action(ann, observation))

            counter += 1

        env.close()

    @staticmethod
    def is_rendering(loops, counter) -> bool:
        if loops is None:
            return True
        else:
            return counter < loops
