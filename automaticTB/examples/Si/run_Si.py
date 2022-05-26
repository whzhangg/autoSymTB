from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.SALCs.decompose import decompose_vectorspace_onelevel, decompose_vectorspace, organize_decomposed_lcs
from automaticTB.MOInteraction import get_free_interaction_AO
from automaticTB.atomic_orbitals import AO
from automaticTB.utilities import print_matrix
import numpy as np
 
from structure import Si_structure

nncluster = Si_structure.nnclusters[0]

from numpy.linalg import matrix_rank, norm

def find_li_vectors(R):
    dim = R.shape[1]
    r = matrix_rank(R) 
    index = np.zeros( r, dtype=int ) #this will save the positions of the li columns in the matrix
    counter = 0
    index[0] = 0 #without loss of generality we pick the first column as linearly independent
    j = 0 #therefore the second index is simply 0

    for i in range(R.shape[0]): #loop over the columns
        if i != j: #if the two columns are not the same
            inner_product = np.dot( R[:,i], R[:,j] ) #compute the scalar product
            norm_i = norm(R[:,i]) #compute norms
            norm_j = norm(R[:,j])

            #inner product and the product of the norms are equal only if the two vectors are parallel
            #therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
            if np.abs(inner_product - norm_j * norm_i) > 1e-6:
                counter += 1 #counter is incremented
                index[counter] = i #index is saved
                j = i #j is refreshed
            #do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!

    R_independent = np.zeros((r, dim))

    i = 0
    #now save everything in a new matrix
    while( i < r ):
        R_independent[i,:] = R[index[i],:] 
        i += 1

    return R_independent

vectorspace = VectorSpace.from_NNCluster(nncluster)
named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
print(len(named_lcs))

free_pairs = get_free_interaction_AO(nncluster, named_lcs, debug=False)
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