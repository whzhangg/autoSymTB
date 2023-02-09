"""This script provide necessary inputs to initialize a tight-binding model"""
import typing

import numpy as np

from automaticTB.solve import interaction
from automaticTB import tools
from automaticTB.properties import tightbinding

def convert_AOpair_to_HijR(aos: typing.List[interaction.AOPairWithValue]):
    hijrs = []
    for aopair in aos:
        p_l = tightbinding.Pindex_lm(
            pindex = aopair.l_AO.primitive_index,
            n = aopair.l_AO.n, l = aopair.l_AO.l, m = aopair.l_AO.m,
            translation = aopair.l_AO.translation
        )
        p_r = tightbinding.Pindex_lm(
            pindex = aopair.r_AO.primitive_index,
            n = aopair.r_AO.n, l = aopair.r_AO.l, m = aopair.r_AO.m,
            translation = aopair.r_AO.translation
        )

        hijrs.append(tightbinding.HijR(p_l, p_r, aopair.value))
    return hijrs

__all__ = ["SingleBandInputs", "get_singleband_tightbinding_with_overlap"]


class SingleBandInputs:
    """
    this class provide the necessary input that will help to provide 
    create a single-band tight-binding model.
    """
    a =  0.0
    b = -1.0
    cell = np.eye(3)
    pos = np.array([[0.5, 0.5, 0.5]])  # atoms in the middle of the cell
    types = [11]                        # Na
    orbitals_nlm = (3, 1, 0)           # 3s orbitals
    home_translations = [(0, 0, 0)]
    all_translations = [
        (0, 0, 0), 
        (1, 0, 0), (-1, 0, 0), 
        (0, 1, 0), (0, -1, 0), 
        (0, 0, 1), (0, 0, -1)
    ]
    def __init__(self, overlap: float = 0.1 ) -> None:
        self._overlap = overlap


    @property
    def interactionPairH(self) -> typing.List[interaction.AOPairWithValue]:
        """
        provide a list of interaction pair Hij
        """
        pairsvalue = []
        values = []
        for h_t in self.home_translations:
            for a_t in self.all_translations:
                home_ao = interaction.AO(
                    equivalent_index = 0,
                    primitive_index = 0,
                    absolute_position = self.pos,
                    translation = h_t,
                    chemical_symbol = tools.chemical_symbols[self.types[0]],
                    n = self.orbitals_nlm[0],
                    l = self.orbitals_nlm[1],
                    m = self.orbitals_nlm[2]
                )
                other_ao = interaction.AO(
                    equivalent_index = 0,
                    primitive_index = 0,
                    translation = a_t,
                    absolute_position = self.pos + a_t,
                    chemical_symbol = tools.chemical_symbols[self.types[0]],
                    n = self.orbitals_nlm[0],
                    l = self.orbitals_nlm[1],
                    m = self.orbitals_nlm[2]
                )
                if home_ao == other_ao:
                    value = self.a
                else:
                    value = self.b

                pairsvalue.append(
                    interaction.AOPairWithValue(home_ao, other_ao, value)
                )

        return pairsvalue


    @property
    def overlapPairS(self) -> typing.List[interaction.AOPairWithValue]:
        """
        provide overlap matrix
        """
        H_pairs = self.interactionPairH
        overlaps = []
        for ao_pair in H_pairs:
            if ao_pair.l_AO == ao_pair.r_AO:
                overlaps.append(interaction.AOPairWithValue(ao_pair.l_AO, ao_pair.r_AO, 1.0))
            else:
                overlaps.append(interaction.AOPairWithValue(ao_pair.l_AO, ao_pair.r_AO, self._overlap))

        return overlaps


def get_singleband_tightbinding_with_overlap(overlap: float) -> tightbinding.TightBindingModel:
    """high level function that return a single band tight binding model"""
    singleband_inputs = SingleBandInputs(overlap)
    HijRs = convert_AOpair_to_HijR(singleband_inputs.interactionPairH)
    SijRs = convert_AOpair_to_HijR(singleband_inputs.overlapPairS)

    return tightbinding.TightBindingModel(
        singleband_inputs.cell, singleband_inputs.pos, singleband_inputs.types,
        HijRs, SijRs
    )