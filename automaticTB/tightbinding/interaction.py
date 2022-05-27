import typing, dataclasses
import numpy as np
from ..utilities import Pair, PairwithValue, tensor_dot


@dataclasses.dataclass
class InteractionPairs:
    pairs: typing.List[Pair]
    interactions: np.ndarray
    """
    this class help to find value in an interaction matrix, we store the data as a list of items
    and we should return the index by giving a pair item1, item2
    the item must have the __equal__ method so that we can use .index()
    """
    def get_index(self, pair: Pair) -> typing.Tuple[int,int]:
        return self.pairs.index(pair)
    
    @classmethod
    def random_from_states(cls, states: list):
        pairs = tensor_dot(states, states)
        coefficients = np.random.random(len(pairs))
        return cls(pairs, coefficients)

    @classmethod
    def zero_from_states(cls, states: list):
        pairs = tensor_dot(states, states)
        coefficients = np.zeros(len(pairs))
        return cls(pairs, coefficients)

    @property
    def HijR(self) -> typing.List[PairwithValue]:
        result = []
        for pair, value in zip(self.pairs, self.interactions):
            result.append([
                pair.left, pair.right, value
            ])
        return result
