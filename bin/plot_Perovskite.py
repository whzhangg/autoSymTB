import typing
from automaticTB.examples import get_perovskite_structure
from automaticTB.interaction import (
    InteractionEquation, AOPair, AOPairWithValue, CombinedInteractionEquation
)
from automaticTB.atomic_orbitals import AO
from automaticTB.printing import print_ao_pairs, print_matrix, print_InteractionPairs
from automaticTB.examples.Perovskite.ao_interaction import (
    get_interaction_values_from_list_AOpairs, get_overlap_values_from_list_AOpairs
)
from automaticTB.tightbinding import TightBindingModel, gather_InteractionPairs_into_HijRs
from automaticTB.structure import Structure
from automaticTB.functions import (
    get_namedLCs_from_nncluster, 
    get_free_AOpairs_from_nncluster_and_namedLCs
)
import numpy as np

cubic_kpos = {
        "M": (np.array([0.5,0.5,0.0]),),
        "X": (np.array([0.5,0.0,0.0]),),
        "G": (np.array([0.0,0.0,0.0]),),
        "R": (np.array([0.5,0.5,0.5]),),
    }

def plot_band(
    structure: Structure, all_interaction_pairs: typing.List[AOPairWithValue], 
    filename: str):
    
    hijRs = gather_InteractionPairs_into_HijRs(all_interaction_pairs)
    perovskiet_TBmodel = TightBindingModel(
        structure.cell, structure.positions, structure.types, hijRs
    )

    for symbol, kpos in cubic_kpos.items():
        print(symbol)
        e, _ = perovskiet_TBmodel.solveE_at_k(kpos)
        to_print = []
        for a_value in e:
            to_print.append( f"{a_value:>9.5f}" )
        print(",".join(to_print))

    from automaticTB.examples.kpaths import cubic_kpath
    from automaticTB.properties.bandstructure import BandStructureResult
    import os

    print("Plot the bandstructure")
    band_result = BandStructureResult.from_tightbinding_and_kpath(perovskiet_TBmodel, cubic_kpath)
    band_result.plot_data(filename)

def remove_same_pair_and_zero_value(pairs_with_value: typing.List[AOPairWithValue]):
    result = []
    for pair in pairs_with_value:
        if np.abs(pair.value) < 1e-3: continue
        #if pair not in result: result.append(pair)
        result.append(pair)
    return result


structure = get_perovskite_structure()
all_named_lcs = []
for nncluster in structure.nnclusters:
    all_named_lcs.append(get_namedLCs_from_nncluster(nncluster))

def previous_way():
    equation_systems: typing.List[InteractionEquation] = []

    for nlc, nncluster in zip(all_named_lcs, structure.nnclusters):
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)

        equation = InteractionEquation.from_nncluster_namedLC(nncluster, nlc)
        equation_systems.append(equation)

    all_interaction_pairs : typing.List[AOPairWithValue] = []
    for equation in equation_systems:
            
        hamiltonian_values = get_interaction_values_from_list_AOpairs(
            structure.cell, structure.positions, equation.free_AOpairs
        )
        
        cluster_interactions = equation.solve_interactions_to_InteractionPairs(hamiltonian_values)

        all_interaction_pairs += cluster_interactions
    return all_interaction_pairs

def current_way():
    equation_systems: typing.List[InteractionEquation] = []

    for nlc, nncluster in zip(all_named_lcs, structure.nnclusters):
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)

        equation = InteractionEquation.from_nncluster_namedLC(nncluster, nlc)
        equation_systems.append(equation)

    combined = CombinedInteractionEquation(equation_systems)
    
    hamiltonian_values = get_interaction_values_from_list_AOpairs(
            structure.cell, structure.positions, combined.free_AOpairs
    )

    return combined.solve_interactions_to_InteractionPairs(hamiltonian_values)

if __name__ == "__main__":
    previous_interaction_pairs = remove_same_pair_and_zero_value(previous_way())
    print_InteractionPairs(structure, previous_interaction_pairs)
    plot_band(structure, previous_interaction_pairs, "previous.pdf")
    current_interaction_pairs = remove_same_pair_and_zero_value(current_way())
    print_InteractionPairs(structure, current_interaction_pairs)
    plot_band(structure, current_interaction_pairs, "current.pdf")