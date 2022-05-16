import typing, abc
import numpy as np

class Interaction(abc.ABC):

    @abc.abstractmethod
    def get_interactions(self) -> np.ndarray:
        pass

class RandomInteraction(Interaction):

    def get_interactions(self) -> np.ndarray:
        pass
