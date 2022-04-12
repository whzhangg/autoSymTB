from symmetry import Symmetry
import torch
from character_table import IrreducibleRP, GroupCharacterTable
from typing import List, Dict
import math
from torch_type import float_type
from atomic_orbital import LinearCombination, CenteredAO, get_dimension, get_linear_combination
from InvariantSpace import VectorSpace, decompose_vectorspace
# a test data, group D3h
# http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=603&option=4


D3h_rotation = [
    Symmetry((0,       0,            0), 0), #0 E        identity
    Symmetry((0,       0,     torch.pi), 1), #1 Sigma_h  horizontal refection
    Symmetry((0,       0, 2*torch.pi/3), 0), #2 C3       120degree rotation
    Symmetry((0,       0,   torch.pi/3), 1), #3 S3       
    Symmetry((0,torch.pi,            0), 0), #4 C2'      rotation around x 
    Symmetry((0,torch.pi,    -torch.pi), 1)  #5 Sigma_v  vertical reflection
]

D3h_class_multiplicity = [1, 1, 2, 2, 3, 3]
D3h_irrepresentations = [
    IrreducibleRP("A1\'", [1, 1, 1, 1, 1, 1]),
    IrreducibleRP("A2\'", [1, 1, 1, 1,-1,-1]),
    IrreducibleRP("A1\"", [1,-1, 1,-1, 1,-1]),
    IrreducibleRP("A2\"", [1,-1, 1,-1,-1, 1]),
    IrreducibleRP("E\'",  [2, 2,-1,-1, 0, 0]),
    IrreducibleRP("E\"",  [2,-2,-1, 1, 0, 0])
]

D3h_table = GroupCharacterTable(
    D3h_rotation, D3h_class_multiplicity, D3h_irrepresentations
)

Cs_rotation = [
    Symmetry((0,       0,            0), 0), #0 E        identity
    Symmetry((0,torch.pi,    -torch.pi), 1)  #5 Sigma_v  vertical reflection
]
all_sym_Cs = [
    [Symmetry((0,       0,            0), 0)], #0 E        identity
    [Symmetry((0,torch.pi,    -torch.pi), 1)]  #5 Sigma_v  vertical reflection
]
Cs_class_multiplicity = [1, 1]
Cs_irrepresentations = [
    IrreducibleRP("A\'", [1, 1]),
    IrreducibleRP("A\"", [1,-1])
]
Cs_table = GroupCharacterTable(
    Cs_rotation, Cs_class_multiplicity, Cs_irrepresentations
)


D3h_positions = torch.tensor([
    [ 2.0,   0.0, 0.0] ,
    [-1.0, math.sqrt(3), 0.0],
    [-1.0,-math.sqrt(3), 0.0],
], dtype = float_type)
D3h_positions = torch.index_select(D3h_positions, 1, torch.tensor([0,2,1]))
D3h_AOs = [ '1x0e' for pos in D3h_positions ]


def find_all_D3h_symmetry_operation(operations: List[torch.Tensor]) -> List[torch.Tensor]:
    # its an ugly method, but maybe the easiest
    equivalence_class = []
    equivalence_class.append([operations[0]])
    equivalence_class.append([operations[1]])
    # 
    s3 = operations[3]
    equivalence_class.append([torch.linalg.matrix_power(s3, 2), torch.linalg.matrix_power(s3, 4)])
    equivalence_class.append([torch.linalg.matrix_power(s3, 1), torch.linalg.matrix_power(s3, 5)]) 
    # s3 * 3 = sigma_h
    c31 = operations[2]
    c32 = torch.linalg.matrix_power(c31, 2)
    equivalence_class.append([
        operations[4], 
        torch.matmul(c32, torch.matmul(operations[4],c31)), 
        torch.matmul(c31, torch.matmul(operations[4],c32))
    ])

    equivalence_class.append([
        operations[5], 
        torch.matmul(c32, torch.matmul(operations[5],c31)), 
        torch.matmul(c31, torch.matmul(operations[5],c32))
    ])

    for o, os in zip(operations, equivalence_class):
        assert torch.allclose(o, os[0])
    return equivalence_class

def find_all_Cs_symmetry_operation(operations: List[torch.Tensor]) -> List[torch.Tensor]:
    # trivial
    return [ [matrix] for matrix in operations ]

def test_generating_subspaces():
    aolist = []
    for pos, ao in zip(D3h_positions, D3h_AOs):
        aolist.append(CenteredAO("C", pos, ao))
    n = get_dimension(aolist)
    lc = get_linear_combination(torch.eye(n, dtype = float_type), aolist)
        
    rep = VectorSpace(lc, D3h_table.symmetries)

    all_sym_D3h = find_all_D3h_symmetry_operation(rep.reducible_matrix)
    subspaces = decompose_vectorspace(rep, D3h_table, all_sym_D3h, normalize=True)

    # decompse E'
    rep_sub = VectorSpace(subspaces["E\'"].LCs, Cs_table.symmetries)
    all_sym_Cs = find_all_Cs_symmetry_operation(rep_sub.reducible_matrix)
    subsubspaces = decompose_vectorspace(rep_sub, Cs_table, all_sym_Cs, normalize=True)


    for name, subspace in subsubspaces.items():
        any_function = any([bool(lc) for lc in subspace.LCs])
        if any_function: 
            print(name)
            print(subspace.rank)
            for lc in subspace.LCs:
                if lc: print(lc)

#test_generating_subspaces()