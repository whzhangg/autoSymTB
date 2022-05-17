import typing, abc
import numpy as np
from ..SALCs.linear_combination import LinearCombination
from .MOcoefficient import SubspaceName, InteractingLCPair

class Interaction(abc.ABC):

    @abc.abstractmethod
    def get_interactions(self, pair: InteractingLCPair) -> float:
        pass

class RandomInteraction(Interaction):

    def get_interactions(self, pair: InteractingLCPair) -> float:
        return np.random.rand()
