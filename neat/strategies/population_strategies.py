from enum import Enum


class PopulationStrategies(Enum):
    CURR_TO_BEST_SPEC = 1
    CURR_TO_RAND_SPEC = 2
    CURR_TO_BEST_INTERSPEC = 3
    CURR_TO_RAND_INTERSPEC = 4

    RAND_TO_RAND_SPEC = 5
    RAND_TO_RAND_INTERSPEC = 6
    RAND_TO_BEST_SPEC = 7
    RAND_TO_BEST_INTERSPEC = 8

    SKIP = 9

    def get_name(self) -> str:
        if self == PopulationStrategies.CURR_TO_BEST_SPEC:
            return "CTB-S"
        if self == PopulationStrategies.CURR_TO_BEST_INTERSPEC:
            return "CTB-I"
        if self == PopulationStrategies.CURR_TO_RAND_SPEC:
            return "CTR-S"
        if self == PopulationStrategies.CURR_TO_RAND_INTERSPEC:
            return "CTR-I"
        if self == PopulationStrategies.RAND_TO_RAND_SPEC:
            return "RTR-S"
        if self == PopulationStrategies.RAND_TO_RAND_INTERSPEC:
            return "RTR-I"
        if self == PopulationStrategies.RAND_TO_BEST_SPEC:
            return "RTB-S"
        if self == PopulationStrategies.RAND_TO_BEST_INTERSPEC:
            return "RTB-I"
