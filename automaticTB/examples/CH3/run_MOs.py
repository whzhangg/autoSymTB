from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_free_interaction_AO
from structure import nncluster
from automaticTB.utilities import Pair
import numpy as np
from automaticTB.utilities import print_matrix
from interactions import get_random_MO_interaction
import string
from automaticTB.atomic_orbitals import AO
 

vectorspace = VectorSpace.from_NNCluster(nncluster)

named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

free_pairs = get_free_interaction_AO(nncluster, named_lcs)
aos = nncluster.AOlist
for pair in free_pairs:
    leftIndex = pair.left
    rightIndex = pair.right
    left: AO = aos[leftIndex]
    right: AO = aos[rightIndex]
    print(" <-> ".join([str(left), str(right)]))
