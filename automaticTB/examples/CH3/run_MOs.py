from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.MOInteraction import get_block_transformer, BlockTransformer, get_free_AO_pairs
from structure import nncluster
from automaticTB.utilities import Pair
import numpy as np
from automaticTB.utilities import print_matrix
from interactions import get_random_MO_interaction
import string

class Simple_mapper:
    def __init__(self) -> None:
        self.memory = {}
        self.used = 0

    
    def get_value(self, name1: str, name2: str):
        if name1 != name2: return 0
        rep = name1.split("->")[0]
        if rep in self.memory:
            return self.memory[rep]
        else:
            chosen = np.random.random()
            self.used += 1
            self.memory[rep] = chosen
            return chosen        
        

vectorspace = VectorSpace.from_NNCluster(nncluster)

named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

free_pairs = get_free_AO_pairs(nncluster, named_lcs)
for pair in free_pairs:
    print(pair.left, pair.right)

raise
conversion_matrices = get_block_transformer(nncluster, named_lcs)
AOs = nncluster.AOlist
MOs = [nlc.name for nlc in named_lcs]

total_number_MO_pair = 0
for i, cm in enumerate(conversion_matrices):
    paired_MO = []
    total_number_MO_pair += len(cm.mo_pairs)
    mapper = Simple_mapper()
    for mopair in cm.mo_pairs:
        rep1 = named_lcs[mopair.left].name
        rep2 = named_lcs[mopair.right].name
        value = mapper.get_value(rep1, rep2)
        #print(value)
        print("< {:>s} | H | {:>s} > = {:>.2f}".format(rep1, rep2, value))
        print()
        paired_MO.append(
            Pair(named_lcs[mopair.left], named_lcs[mopair.right])
        )
    print(cm.matrix.real)
    

print(total_number_MO_pair)