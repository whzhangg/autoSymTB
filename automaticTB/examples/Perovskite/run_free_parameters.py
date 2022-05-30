import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_free_interaction_AO
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.printing import print_ao_pairs, print_matrix

# seems to be a bit problematic
def run_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    for i, nncluster in enumerate(structure.nnclusters):
        #if i == 0: continue
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        print("Solve Interaction ...")
        free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
        
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
        free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=True)
            
        print_ao_pairs(nncluster, free_pairs)
        print()

if __name__ == "__main__":
    run_perovskite_SALCs()