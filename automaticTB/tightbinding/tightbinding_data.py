import dataclasses, typing
import numpy as np

@dataclasses.dataclass
class Pindex_lm:
    pindex: int  # index in primitive cell
    l: int
    m: int
    translation: np.ndarray

    def __eq__(self, other) -> bool:
        return self.pindex == other.pindex and \
               self.l == other.l and self.m == other.m and \
               np.allclose(self.translation, other.translation)


@dataclasses.dataclass
class HijR:
    left: Pindex_lm
    right: Pindex_lm
    value: float

@dataclasses.dataclass
class TBData:
    cell: np.ndarray
    positions: np.ndarray
    types: typing.List[int]
    HijR_list: typing.List[HijR]