import typing, abc
import numpy as np
from .MOcoefficient import BraKet


class Interaction(abc.ABC):

    @abc.abstractmethod
    def get_interactions(self, pair: BraKet) -> float:
        pass


class RandomInteraction(Interaction):
    def __init__(self) -> None:
        self._stored_interaction = {}

    def get_interactions(self, pair: BraKet) -> float:
        if pair.bra.irreps != pair.ket.irreps:
            return 0.0
        
        random_value = np.random.rand()
        return random_value
