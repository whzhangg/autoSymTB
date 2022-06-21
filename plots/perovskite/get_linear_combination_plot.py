import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.structure import NearestNeighborCluster, Structure
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.printing import print_ao_pairs, print_matrix, print_InteractionPairs
from automaticTB.examples.Perovskite.ao_interaction import get_interaction_values_from_list_AOpairs
from automaticTB.plotting import DensityCubePlot
from automaticTB.tools import write_yaml

named_to_print = [
    "A_1g @ 1^th A_1g",
    "A_1g @ 2^th A_1g",
    "A_1g @ 3^th A_1g",
    "T_1u->A_1 @ 1^th T_1u",
    "T_1u->A_1 @ 2^th T_1u",
    "T_1u->A_1 @ 3^th T_1u",
    "T_1u->A_1 @ 4^th T_1u",
    "E_g->A_1 @ 2^th E_g",
    "E_g->A_2 @ 2^th E_g",
    "T_1g->A_2 @ 1^th T_1g",
    "T_2u->B_1 @ 1^th T_2u",
    "T_2g->B_1 @ 1^th T_2g"
]

# seems to be a bit problematic
def run_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    for i, nncluster in enumerate(structure.nnclusters):
        if i != 0: continue
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        for nlc in named_lcs:
            if str(nlc.name) in named_to_print:
                plot = DensityCubePlot.from_linearcombinations([nlc.lc], "high")
                plot.plot_to_file(f"{str(nlc.name)}.cube")

        print()
        print("Solve Interaction ...")
        free_pairs = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs).free_AOpairs
        
        print_ao_pairs(nncluster, free_pairs)
        print()

def test_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    nnclusters = (structure.nnclusters[1],structure.nnclusters[2])
    for nncluster in nnclusters:
        #for seitz, op in nncluster.sitesymmetrygroup.dressed_op.items():
        #    print(seitz)
        #    print(print_matrix(op))
        #for irrep, value in nncluster.sitesymmetrygroup.irreps.items():
        #    print(irrep)
        #    print(value)
            
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        for nlcs in named_lcs:
            print(nlcs.name)
            print(nlcs.lc)
            print("")
        print("Solve Interaction ...")
        equation_system = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        free_pairs = equation_system.free_AOpairs
        interactions = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_pairs)
        all_interactions = equation_system.solve_interactions_to_InteractionPairs(interactions)
        print_InteractionPairs(nncluster, all_interactions)
        print()

if __name__ == "__main__":
    run_perovskite_SALCs()