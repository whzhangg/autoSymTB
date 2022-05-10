# find the basis functions for a vector space
from atomic_orbital import VectorSpace, LinearCombination, VectorSpacewithBasis
from group_subgroup import GroupTable
from typing import Dict, List, Tuple
import torch


def find_SALCs(vector_space: VectorSpace, group: str) -> VectorSpacewithBasis:
    input_rank = vector_space.rank
    nested_subspaces = find_salc(vector_space, GroupTable(group), normalize=True)
    all_vector = get_nested_nonzero_vectorspace(nested_subspaces)
    basises = [ ]
    names = [ ]

    for n, b in all_vector:
        reduced = b.remove_linear_dependent()
        if n == "E'->A'":
            from plot_mo import plot_density
            for i,lc in enumerate(reduced.LCs):
                print(lc)
                xs, ys, zs, density = lc.density()
                plot_density(xs, ys, zs, density, f"{i+1}-{n}.html")
        for lc in reduced.LCs[0:reduced.rank]:
            basises.append(lc)
            names.append(n)
    #for n, basis in zip(names, basises):
    #    print(n, basis)
    assert input_rank == len(basises), (input_rank, len(basises))
    return VectorSpacewithBasis(basises, names)


def get_nested_nonzero_vectorspace(space:dict) -> List[Tuple[str,VectorSpace]]:
    all_vector_space = []
    for key, value in space.items():
        if type(value) == dict:
            sub = get_nested_nonzero_vectorspace(value)
            all_vector_space += [ (f"{key}->{n}", v) for n, v in sub ]
        else:
            if value.rank > 0:
                all_vector_space.append((key, value))
    return all_vector_space


def find_salc(vector_space: VectorSpace, group: GroupTable, normalize: bool = True) -> Dict[str, VectorSpace]:
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
            subspaces[irrep.symbol] = find_salc(subspaces[irrep.symbol], group.get_subgroup())
    return subspaces


def find_subspace(vector_space: VectorSpace, group: str, normalize: bool = False) -> Dict[str, VectorSpace]:
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
    pass