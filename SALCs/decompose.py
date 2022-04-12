# find the basis functions for a vector space
from atomic_orbital import VectorSpace, LinearCombination
from symmetry import Symmetry
from group_subgroup import GroupTable
from typing import Tuple, Dict, List, Union
import torch
from torch_type import float_type


def print_subspace(subspace_dict: Dict[str, VectorSpace]):

    for name, subspace in subspace_dict.items():
        any_function = any([bool(lc) for lc in subspace.LCs])
        if any_function: 
            print(name)
            print(subspace.rank)
            for lc in subspace.LCs:
                if lc: print(lc)


def find_SALC(vector_space: VectorSpace, group: GroupTable, normalize: bool = True) -> Dict[str, VectorSpace]:

    subspaces = {}
    for irrep in group.irreps:
        transformed_LCs = []
        for lc in vector_space.LCs:
            sum_coefficients = torch.zeros_like(lc.coefficient, dtype = float)
            for isym, sym in enumerate(group.symmetries):
                rotated = sym.rotate_LC(lc)
                sum_coefficients += rotated.coefficient * irrep.characters[isym] * irrep.dimension / group.order

            transformed_LCs.append(LinearCombination(sum_coefficients, lc.AOs , normalize))
        subspaces[irrep.symbol] = VectorSpace(transformed_LCs)

    for irrep in group.irreps:
        if subspaces[irrep.symbol].rank > 1 and irrep.dimension > 1 and group.get_subgroup():
            subspaces[irrep.symbol] = find_SALC(subspaces[irrep.symbol], group.get_subgroup())
    return subspaces


def decompose(vector_space: VectorSpace, group: str, normalize: bool = False) -> Dict[str, VectorSpace]:
    group_table = GroupTable(group)
    subspaces = {}
    for irrep in group_table.irreps:
        transformed_LCs = []
        for lc in vector_space.LCs:
            sum_coefficients = torch.zeros_like(lc.coefficient, dtype = float)
            for isym, sym in enumerate(group_table.symmetries):
                rotated = sym.rotate_LC(lc)
                sum_coefficients += rotated.coefficient * irrep.characters[isym] * irrep.dimension / group_table.order

            transformed_LCs.append(LinearCombination(sum_coefficients, lc.AOs , normalize))
        subspaces[irrep.symbol] = VectorSpace(transformed_LCs)
    return subspaces
    
if __name__ == "__main__":
    from atomic_orbital import CenteredAO, get_coefficient_dimension, get_linear_combination
    import math
    D3h_positions = torch.tensor([
        [ 2.0,   0.0, 0.0] ,
        [-1.0, math.sqrt(3), 0.0],
        [-1.0,-math.sqrt(3), 0.0],
    ], dtype = float_type)
    D3h_positions = torch.index_select(D3h_positions, 1, torch.tensor([0,2,1]))
    D3h_AOs = [ '1x0e' for pos in D3h_positions ]
    aolist = []
    for i, (pos, ao) in enumerate(zip(D3h_positions, D3h_AOs)):
        aolist.append(CenteredAO(f"C{i+1}", pos, ao))
    n = get_coefficient_dimension(aolist)
    lc = get_linear_combination(torch.eye(n, dtype = float_type), aolist)
        
    rep = VectorSpace(lc)
    result = decompose(rep, "D3h")
    #print(result)
    #result = decompose(result["E\'"], "Cs")
    #print(result)

    result = find_SALC(rep, GroupTable("D3h"))
    print(result)