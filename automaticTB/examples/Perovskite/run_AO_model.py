"""
we obtain bandstructure using atomic interaction directly
"""

import numpy as np
import os, typing

from automaticTB.structure.structure import Structure
from automaticTB.tightbinding.tightbinding_model import TightBindingModel, make_HijR_list
from automaticTB.tools import find_RCL
from automaticTB.printing import print_matrix
from automaticTB.tools import InteractionPairs

from structure import cell, types, positions, orbits_used
from ao_interaction import get_interaction_values_from_list_AOpairs

structure: Structure = Structure.from_cpt_rcut(cell, positions, types, orbits_used, 3.0)  
reciprocal_cell = find_RCL(structure.cell)
# distance Pb - Cl is ~2.8 A

def get_structure() -> Structure:
    return Structure.from_cpt_rcut(cell, positions, types, orbits_used, 3.0)


def get_AO_interactions(structure: Structure):
    ao_pairs = []
    interactions = []
    for cluster in structure.nnclusters:
        for subspace_pair in cluster.subspace_pairs:
            ao_pairs_to_do = cluster.get_subspace_AO_Pairs(subspace_pair)
            value = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, ao_pairs_to_do)
            ao_pairs += ao_pairs_to_do
            interactions += value
    return InteractionPairs(ao_pairs, interactions)


def obtain_and_plot_bandstructure(model: TightBindingModel, filename: str = "HalidePerovskite_band.pdf") -> None:
    from automaticTB.properties.bandstructure import Kline, Kpath, get_bandstructure_result

    # as in the paper of 
    # Symmetry-Based Tight Binding Modeling of Halide Perovskite Semiconductors
    kpos = {
        "M": np.array([0.5,0.5,0.0]),
        "X": np.array([0.5,0.0,0.0]),
        "G": np.array([0.0,0.0,0.0]),
        "R": np.array([0.5,0.5,0.5]),
    }

    lines = [
        Kline("M", kpos["M"], "R", kpos["R"], 10),
        Kline("R", kpos["R"], "G", kpos["G"], 18),
        Kline("G", kpos["G"], "X", kpos["X"], 10),
        Kline("X", kpos["X"], "M", kpos["M"], 10),
        Kline("M", kpos["M"], "G", kpos["G"], 15),
    ]

    kpath = Kpath(reciprocal_cell , lines)

    band_result = get_bandstructure_result(model, kpath)
    band_result.plot_data(filename)


def obtain_and_plot_dos(model: TightBindingModel, filename: str = "HalidePerovskite_dos.pdf") -> None:
    from automaticTB.properties.dos import get_tetrados_result
    tetrados = get_tetrados_result(model, ngrid=np.array([10,10,10]))
    tetrados.plot_data(filename)


if __name__ == "__main__":
    structure = get_structure()
    interactions = get_AO_interactions(structure)
    HijRlist = make_HijR_list([interactions])
    for HijR in HijRlist:
        print(HijR)
    model = TightBindingModel(structure.cell, structure.positions, structure.types, HijRlist)
    obtain_and_plot_bandstructure(model)