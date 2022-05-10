from .vectorspace import VectorSpace
from ..linear_combination import LinearCombination, Site
from DFTtools.SiteSymmetry.site_symmetry_group import SiteSymmetryGroup
import typing
import numpy as np
from ..rotation import rotate_linear_combination_from_symmetry_matrix


def find_salc(vector_space: VectorSpace, group, normalize: bool = True):
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

def decompose_vectorspace(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.Dict[str, VectorSpace]:
    subspaces = {}
    group_order = len(group.operations)
    for irrep, characters in group.irreps.items():
        irrep_dimension = group.irrep_dimension[irrep]
        transformed_LCs = []
        for lc in vectorspace.get_linear_combinations():
            sum_coefficients = np.zeros_like(lc.coefficients)
            for op, chi in zip(group.operations, characters):
                rotated = rotate_linear_combination_from_symmetry_matrix(lc, op)
                rotated.scale_coefficients(chi * irrep_dimension / group_order)
                sum_coefficients += rotated.coefficients
            transformed_LCs.append(LinearCombination(vectorspace.sites, sum_coefficients))
        subspaces[irrep] = VectorSpace.from_list_of_linear_combinations(transformed_LCs)
    return subspaces
