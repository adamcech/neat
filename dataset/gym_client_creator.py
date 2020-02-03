from dataset.gym_client import GymClient


class GymClientCreator:

    @staticmethod
    def create_cart_pole(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 5 if max_trials is None else max_trials
        max_episodes = 200 if max_episodes is None else max_episodes

        return GymClient("CartPole-v0", True, 4, 2, bias, max_trials, max_episodes)

    @staticmethod
    def create_acrobot(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 2 if max_trials is None else max_trials
        max_episodes = 500 if max_episodes is None else max_episodes

        return GymClient("Acrobot-v1", True, 6, 3, bias, max_trials, max_episodes)

    @staticmethod
    def create_lunar_lander(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 3 if max_trials is None else max_trials
        max_episodes = 500 if max_episodes is None else max_episodes

        return GymClient("LunarLander-v2", True, 8, 4, bias, max_trials, max_episodes)

    @staticmethod
    def create_lunar_lander_continuous(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 3 if max_trials is None else max_trials
        max_episodes = 500 if max_episodes is None else max_episodes

        return GymClient("LunarLanderContinuous-v2", False, 8, 2, bias, max_trials, max_episodes)

    @staticmethod
    def create_bipedal_walker(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 1 if max_trials is None else max_trials
        max_episodes = 1600 if max_episodes is None else max_episodes

        return GymClient("BipedalWalker-v2", False, 24, 4, bias, max_trials, max_episodes)

    @staticmethod
    def create_bipedal_walker_hardcore(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 1 if max_trials is None else max_trials
        max_episodes = 2000 if max_episodes is None else max_episodes

        return GymClient("BipedalWalkerHardcore-v2", False, 24, 4, bias, max_trials, max_episodes)

    @staticmethod
    def create_pacman(bias: int = None, max_trials: int = None, max_episodes: int = None):
        bias = 0 if bias is None else bias
        max_trials = 1 if max_trials is None else max_trials
        max_episodes = 10000 if max_episodes is None else max_episodes

        return GymClient("MsPacman-ram-v0", True, 128, 9, bias, max_trials, max_episodes)
