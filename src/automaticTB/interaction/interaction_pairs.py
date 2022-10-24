import typing, dataclasses
import numpy as np
from ..tools import Pair, PairwithValue
from ..atomic_orbitals import AO 
from ..parameters import zero_tolerance

__all__ = ["AOPair", "AOPairWithValue"]


@dataclasses.dataclass
class AOPair:
    """
    just a pair of AO that enable comparison, AOs can be compared directly
    """
    l_AO: AO
    r_AO: AO

    @property
    def left(self) -> AO:
        return self.l_AO
    
    @property
    def right(self) -> AO:
        return self.r_AO

    @classmethod
    def from_pair(cls, aopair: Pair) -> "AOPair":
        return cls(aopair.left, aopair.right)

    def as_pair(self) -> Pair:
        return Pair(self.l_AO, self.r_AO)

    def __eq__(self, o: "AOPair") -> bool:
        return  (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
                (self.l_AO == o.r_AO and self.r_AO == o.l_AO)


@dataclasses.dataclass
class AOPairWithValue(AOPair):
    value: typing.Union[float, complex]

    def __eq__(self, o: "AOPairWithValue") -> bool:
        valueOK = np.isclose(self.value, o.value, atol = zero_tolerance)
        pair_OK = (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
                  (self.l_AO == o.r_AO and self.r_AO == o.l_AO)
        
        return  valueOK and pair_OK


@dataclasses.dataclass
class InteractionPairs:
    pairs: typing.List[AOPair]
    interactions: np.ndarray
    """
    this class help to find value in an interaction matrix, we store the data as a list of items
    and we should return the index by giving a pair item1, item2
    the item must have the __equal__ method so that we can use .index()
    """
    def get_index(self, pair: Pair) -> typing.Tuple[int,int]:
        return self.pairs.index(pair)

    @property
    def AO_energies(self) -> typing.List[PairwithValue]:
        """
        return a list of (AO1, AO2, interaction value)
        """
        result = []
        for pair, value in zip(self.pairs, self.interactions):
            result.append(
                PairwithValue(pair.left, pair.right, value)
            )
        return result
