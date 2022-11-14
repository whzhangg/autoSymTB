import typing, dataclasses
import numpy as np
from ...solve.interaction import AO, AOPairWithValue
from ...parameters import zero_tolerance

"""
this module defines the interface between the result of the code to create tight-binding model
"""

__all__ = [
    "Pindex_lm", "HijR", "SijR", 
    "gather_InteractionPairs_into_HijRs",
    "gather_InteractionPairs_into_SijRs"
]

@dataclasses.dataclass
class Pindex_lm:
    pindex: int  # index in primitive cell
    n: int
    l: int
    m: int
    translation: np.ndarray

    def __eq__(self, other: "Pindex_lm") -> bool:
        return self.pindex == other.pindex and \
               self.n == other.n and \
               self.l == other.l and self.m == other.m and \
               np.allclose(self.translation, other.translation, atol = zero_tolerance)


@dataclasses.dataclass
class HijR:
    left: Pindex_lm
    right: Pindex_lm
    value: float


@dataclasses.dataclass
class SijR:
    left: Pindex_lm
    right: Pindex_lm
    value: float


def gather_InteractionPairs_into_HijRs(AOinteractions: typing.List[AOPairWithValue]) \
-> typing.List[HijR]:
    HijR_list: typing.List[HijR] = []
    for aopairvalue in AOinteractions:
        left: AO = aopairvalue.l_AO
        right: AO = aopairvalue.r_AO
        HijR_list.append(
            HijR(
                Pindex_lm(left.primitive_index, left.n, left.l, left.m, left.translation),
                Pindex_lm(right.primitive_index, right.n, right.l, right.m, right.translation),
                aopairvalue.value
            )
        )
    return HijR_list


def gather_InteractionPairs_into_SijRs(AOinteractions: typing.List[AOPairWithValue]) \
-> typing.List[SijR]:
    SijR_list: typing.List[HijR] = []
    for aopairvalue in AOinteractions:
        left: AO = aopairvalue.l_AO
        right: AO = aopairvalue.r_AO
        SijR_list.append(
            HijR(
                Pindex_lm(left.primitive_index, left.n, left.l, left.m, left.translation),
                Pindex_lm(right.primitive_index, right.n, right.l, right.m, right.translation),
                aopairvalue.value
            )
        )
    return SijR_list

