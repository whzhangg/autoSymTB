import typing, abc
import numpy as np
from ..SALCs.linear_combination import LinearCombination
from .MOcoefficient import SubspaceName, InteractingLCPair

class Interaction(abc.ABC):

    @abc.abstractmethod
    def get_interactions(self, pair: InteractingLCPair) -> float:
        pass

def interaction_zero(pair: InteractingLCPair):
    name1 = pair.rep1
    name2 = pair.rep2
    if name1.irreps != name2.irreps:
        return True


class RandomInteraction(Interaction):
    def __init__(self) -> None:
        self._stored_interaction = {}

    def get_interactions(self, pair: InteractingLCPair) -> float:
        if pair.rep1.irreps != pair.rep2.irreps:
            return 0.0
        
        random_value = np.random.rand()
        return random_value
