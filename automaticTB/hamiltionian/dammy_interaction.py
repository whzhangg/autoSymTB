import typing, abc
import numpy as np
from .lc_product import ListofNamedLC

class Interaction(abc.ABC):
    def __init__(self, list_lc: ListofNamedLC) -> None:
        self.listLC = list_lc

    @abc.abstractmethod
    def get_interactions(self) -> np.ndarray:
        pass

class RandomInteraction(Interaction):

    def get_interactions(self) -> np.ndarray:
        pass
