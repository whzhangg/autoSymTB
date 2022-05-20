from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_block_transformer, BlockTransformer
from structure import nncluster
from automaticTB.utilities import Pair
import numpy as np
from automaticTB.utilities import print_matrix
from interactions import get_random_MO_interaction

vectorspace = VectorSpace.from_NNCluster(nncluster)

named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

conversion_matrices = get_block_transformer(nncluster, named_lcs)
AOs = nncluster.AOlist
MOs = [nlc.name for nlc in named_lcs]

total_number_MO_pair = 0
for i, cm in enumerate(conversion_matrices):
    paired_MO = []
    total_number_MO_pair += len(cm.mo_pairs)
    for mopair in cm.mo_pairs:
        print(named_lcs[mopair.left].name)
        print(named_lcs[mopair.left].lc)
        print(named_lcs[mopair.right].name)
        print(named_lcs[mopair.right].lc)
        print()
        paired_MO.append(
            Pair(named_lcs[mopair.left], named_lcs[mopair.right])
        )
    mointer = get_random_MO_interaction(paired_MO)
    aointer = cm.inv_matrix.dot(mointer)
    for aopair,value in zip(cm.ao_pairs, aointer):
        left = AOs[aopair.left]
        right = AOs[aopair.right]
        #print(left.chemical_symbol, left.l, left.m, 
        #    right.chemical_symbol, right.l, right.m, )
        #print(value)

print(total_number_MO_pair)