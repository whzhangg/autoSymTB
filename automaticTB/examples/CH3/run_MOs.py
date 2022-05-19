from automaticTB.SALCs.vectorspace import VectorSpace
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.hamiltionian.MOInteraction import get_rep_LC_from_decomposed_vectorspace_as_dict, get_conversion_matrices, ConversionMatrix
from structure import nncluster
from automaticTB.utilities import Pair
import numpy as np
from automaticTB.utilities import print_matrix
from interactions import get_random_MO_interaction

vectorspace = VectorSpace.from_NNCluster(nncluster)
decomposed = decompose_vectorspace(vectorspace, nncluster.sitesymmetrygroup)
named_lcs = get_rep_LC_from_decomposed_vectorspace_as_dict(decomposed)

conversion_matrices = get_conversion_matrices(nncluster, named_lcs)
AOs = nncluster.AOlist
MOs = [nlc.name for nlc in named_lcs]

for i, cm in enumerate(conversion_matrices):
    paired_MO = []
    for mopair in cm.mo_pairs:
        paired_MO.append(
            Pair(named_lcs[mopair.left], named_lcs[mopair.right])
        )
    mointer = get_random_MO_interaction(paired_MO)
    aointer = cm.inv_matrix.dot(mointer)
    for aopair,value in zip(cm.ao_pairs, aointer):
        left = AOs[aopair.left]
        right = AOs[aopair.right]
        print(left.chemical_symbol, left.l, left.m, 
            right.chemical_symbol, right.l, right.m, )
        print(value)