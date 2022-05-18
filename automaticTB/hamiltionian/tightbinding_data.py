import dataclasses, typing
import numpy as np
from .MOcoefficient import AO, InteractionMatrix

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


def make_HijR_list(AOinteractions: typing.List[InteractionMatrix]) -> typing.List[HijR]:
    HijR_list: typing.List[HijR] = []
    for interaction in AOinteractions:
        for pair, value in zip(interaction.flattened_braket, interaction.flattened_interaction):
            left: AO = pair.bra
            right: AO = pair.ket
            if left.at_origin == True:
                HijR_list.append(
                    HijR(
                        Pindex_lm(left.primitive_index, left.l, left.m, left.translation),
                        Pindex_lm(right.primitive_index, right.l, right.m, right.translation),
                        value
                    )
                )
    return HijR_list


def generate_tightbinding_data(
    cell: np.ndarray, 
    positions: typing.List[np.ndarray],
    types: typing.List[int],
    AOinteractions: typing.List[InteractionMatrix]
):
    HijR_list: typing.List[HijR] = []
    for interaction in AOinteractions:
        for pair, value in zip(interaction.flattened_braket, interaction.flattened_interaction):
            left: AO = pair.bra
            right: AO = pair.ket
            if left.at_origin == True:
                HijR_list.append(
                    HijR(
                        Pindex_lm(left.primitive_index, left.l, left.m, left.translation),
                        Pindex_lm(right.primitive_index, right.l, right.m, right.translation),
                        value
                    )
                )

    return TBData(cell, positions, types, HijR_list)