"""This script provide necessary inputs to initialize a tight-binding model"""
import numpy as np
import typing
from ...interaction import AO, AOPairWithValue
from ...tools import chemical_symbols
from ..tightbinding_model import TightBindingModel
from ..hij import gather_InteractionPairs_into_HijRs, gather_InteractionPairs_into_SijRs


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
    def interactionPairH(self) -> typing.List[AOPairWithValue]:
        """
        provide a list of interaction pair Hij
        """
        pairsvalue = []
        values = []
        for h_t in self.home_translations:
            for a_t in self.all_translations:
                home_ao = AO(
                    cluster_index = 0, 
                    equivalent_index = 0,
                    primitive_index = 0,
                    absolute_position = self.pos,
                    translation = h_t,
                    chemical_symbol = chemical_symbols[self.types[0]],
                    n = self.orbitals_nlm[0],
                    l = self.orbitals_nlm[1],
                    m = self.orbitals_nlm[2]
                )
                other_ao = AO(
                    cluster_index = 0, 
                    equivalent_index = 0,
                    primitive_index = 0,
                    translation = a_t,
                    absolute_position = self.pos + a_t,
                    chemical_symbol = chemical_symbols[self.types[0]],
                    n = self.orbitals_nlm[0],
                    l = self.orbitals_nlm[1],
                    m = self.orbitals_nlm[2]
                )
                if home_ao == other_ao:
                    value = self.a
                else:
                    value = self.b

                pairsvalue.append(
                    AOPairWithValue(home_ao, other_ao, value)
                )

        return pairsvalue


    @property
    def overlapPairS(self) -> typing.List[AOPairWithValue]:
        """
        provide overlap matrix
        """
        H_pairs = self.interactionPairH
        overlaps = []
        for ao_pair in H_pairs:
            if ao_pair.l_AO == ao_pair.r_AO:
                overlaps.append(AOPairWithValue(ao_pair.l_AO, ao_pair.r_AO, 1.0))
            else:
                overlaps.append(AOPairWithValue(ao_pair.l_AO, ao_pair.r_AO, self._overlap))

        return overlaps


def get_singleband_tightbinding_with_overlap(overlap: float) -> TightBindingModel:
    """high level function that return a single band tight binding model"""
    singleband_inputs = SingleBandInputs(overlap)
    HijRs = gather_InteractionPairs_into_HijRs(singleband_inputs.interactionPairH)
    SijRs = gather_InteractionPairs_into_SijRs(singleband_inputs.overlapPairS)

    return TightBindingModel(
        singleband_inputs.cell, singleband_inputs.pos, singleband_inputs.types,
        HijRs, SijRs
    )