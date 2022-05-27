from automaticTB.SALCs import VectorSpace
from automaticTB.SALCs.decompose_new import decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_free_interaction_AO
from automaticTB.atomic_orbitals import AO
from automaticTB.utilities import print_matrix
import numpy as np
from automaticTB.utilities import load_yaml, save_yaml
 
from structure import Si_structure

nncluster = Si_structure.nnclusters[0]

vectorspace = VectorSpace.from_NNCluster(nncluster)
#named_lcs = decompose_vectorspace_onelevel_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
print(len(named_lcs))

free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
print(free_pairs)
aos = nncluster.AOlist
for pair in free_pairs:
    leftIndex = pair.left
    rightIndex = pair.right
    left: AO = aos[leftIndex]
    right: AO = aos[rightIndex]
    print(" <-> ".join([str(left), str(right)]))

"""
    vs_non_zeros = vs.get_nonzero_linear_combinations()
    if vs_non_zeros:
        organized = organize_decomposed_lcs(vs_non_zeros, nncluster.sitesymmetrygroup, nncluster.sitesymmetrygroup.irrep_dimension[name])
        assert vs.rank == len(vs_non_zeros)
        for subvs in organized:
            for lc in subvs:
                print("-------")
                print(lc)
            print("-----------------")
free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=True)
aos = nncluster.AOlist
for pair in free_pairs:
    leftIndex = pair.left
    rightIndex = pair.right
    left: AO = aos[leftIndex]
    right: AO = aos[rightIndex]
    print(" <-> ".join([str(left), str(right)]))

"""