import typing, dataclasses
import numpy as np
from ...parameters import ztol

"""
this module defines the interface between the result of the code to create tight-binding model
"""

__all__ = [
    "Pindex_lm", "HijR", "SijR"
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
               np.allclose(self.translation, other.translation, atol = ztol)


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

