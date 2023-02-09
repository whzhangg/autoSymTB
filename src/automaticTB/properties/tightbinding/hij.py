import dataclasses

import numpy as np

@dataclasses.dataclass
class Pindex_lm:
    """unique index of an orbit in solids"""
    pindex: int  
    n: int
    l: int
    m: int
    translation: np.ndarray

    def __eq__(self, other: "Pindex_lm") -> bool:
        return self.pindex == other.pindex and \
               self.n == other.n and \
               self.l == other.l and self.m == other.m and \
               np.allclose(self.translation, other.translation)


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

