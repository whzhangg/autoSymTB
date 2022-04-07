from ao_representation import D3h_AOs, AORepresentation
from character_table import D3h_table, GroupCharacterTable
from torch_type import float_type
import torch
from typing import List
from basisfunction import LinearCombination

def find_all_symmetry_operation(operations: List[torch.Tensor]) -> List[torch.Tensor]:
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

def equivalence_class(operation: torch.Tensor, operations: List[torch.Tensor]) -> List[torch.Tensor]:
    """it will find a list of conjugate symmetry operation"""
    result = [operation]

def torch_round(input:torch.Tensor, decimals: int) -> torch.Tensor:
    return torch.round(input * 10**decimals) / (10**decimals)

def find_basis(
    AO_rep: AORepresentation,
    table: GroupCharacterTable,
    all_symmetry: List[torch.Tensor]
) -> torch.Tensor:
    """
    it will output basis functions
    """
    coefficients = table.decompose_representation(AO_rep.reducible_trace)  
    basis_function_name = AO_rep.bfs
    for name, coeff in coefficients.items():
        characters = table.irreps_characters[name]
        dimension = characters[0] # dimension of irrep
        untouched_basis = torch.eye(AO_rep.nbfs, dtype = float_type)
        new_basis = torch.zeros_like(untouched_basis, dtype = float_type)
        for isym, _ in enumerate(AO_rep.reducible_matrix):
            for full_matrix in all_symmetry[isym]:
                for i, bf in enumerate(untouched_basis):
                    new_basis[i] += torch.matmul(full_matrix, bf) * characters[isym] * dimension / table.order
        #new_basis = torch_round(new_basis, 15)
        print(name, coeff)
        for nb in new_basis:
            name = str(LinearCombination(nb, basis_function_name).normalized())
            if name: print(name)
        print(f"Matrix rank: {torch.linalg.matrix_rank(new_basis, tol = 1e-10).item()}")

# we should rewrite the class so that the functions can be transformed conveniently

all_sym = find_all_symmetry_operation(D3h_AOs.reducible_matrix)
find_basis(D3h_AOs, D3h_table, all_sym)
