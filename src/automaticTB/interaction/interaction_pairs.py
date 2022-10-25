import typing, dataclasses
import numpy as np
from ..tools import Pair
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
    """AOPair with value additionally associated with it, overwrite the __eq__ method"""
    value: typing.Union[float, complex]

    def __eq__(self, o: "AOPairWithValue") -> bool:
        valueOK = np.isclose(self.value, o.value, atol = zero_tolerance)
        pair_OK = (self.l_AO == o.l_AO and self.r_AO == o.r_AO) or \
                  (self.l_AO == o.r_AO and self.r_AO == o.l_AO)
        
        return  valueOK and pair_OK

