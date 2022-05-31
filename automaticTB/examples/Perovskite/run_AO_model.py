import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation, InteractionPairs
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.printing import print_ao_pairs, print_matrix, print_InteractionPairs
from automaticTB.examples.Perovskite.ao_interaction import get_interaction_values_from_list_AOpairs
from automaticTB.tightbinding import TightBindingModel, gather_InteractionPairs_into_HijRs
from automaticTB.tools import find_RCL

# seems to be a bit problematic
def get_tightbinding_model():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    interaction_clusters = []

    for i, nncluster in enumerate(structure.nnclusters):
        #if i == 0: continue
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        print("Solve Interaction ...")
        equation_system = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        free_pairs = equation_system.free_AOpairs
        values = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_pairs)
        all_interactions = equation_system.solve_interactions_to_InteractionPairs(values)
        interaction_clusters.append(all_interactions)
        # self interaction
        self_pairs = nncluster.centered_selfinteraction_pairs
        values = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, self_pairs)
        ipairs = InteractionPairs(self_pairs, values)
        interaction_clusters.append(ipairs)

    hijRs = gather_InteractionPairs_into_HijRs(interaction_clusters)
    return TightBindingModel(structure.cell, structure.positions, structure.types, hijRs)


def obtain_and_plot_bandstructure(model: TightBindingModel, filename: str = "HalidePerovskite_band.pdf") -> None:
    from automaticTB.properties.bandstructure import Kline, Kpath, get_bandstructure_result
    cell = model.cell
    reciprocal_cell = find_RCL(cell)

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
    model = get_tightbinding_model()
    obtain_and_plot_bandstructure(model)