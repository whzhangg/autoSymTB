import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_free_interaction_AO
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.examples.utilities import print_ao_pairs

# seems to be a bit problematic
def run_perovskite_SALCs():
    structure = get_perovskite_structure()
    print("Solving for the free nearest neighbor interaction in Perovskite")
    print("Starting ...")
    for nncluster in structure.nnclusters:
        print("Cluster centered on " + str(nncluster.crystalsites[nncluster.origin_index]))
        print(nncluster.sitesymmetrygroup.groupname)
        vectorspace = VectorSpace.from_NNCluster(nncluster)
        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        print("Solve Interaction ...")
        free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
        
        print_ao_pairs(nncluster.AOlist, free_pairs)
        print()

if __name__ == "__main__":
    run_perovskite_SALCs()