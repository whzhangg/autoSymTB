import typing
import numpy as np
from automaticTB.sitesymmetry.site_symmetry_group import SiteSymmetryGroup
from .vectorspace import VectorSpace
from .subduction import subduction_data
from .linear_combination import LinearCombination


class NamedLC(typing.NamedTuple):
    name: str
    lc: LinearCombination


def decompose_vectorspace_to_namedLC(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.List[NamedLC]:
    decomposed = decompose_vectorspace(vectorspace, group)
    return get_rep_LC_from_decomposed_vectorspace_as_dict(decomposed)


def decompose_vectorspace(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.Dict[str, VectorSpace]:
    """the return is a dictionary indexed by the irreducible representation"""
    recursive_result = decompose_vectorspace_recursive(vectorspace, group)
    return get_nested_nonzero_vectorspace(recursive_result)


def get_rep_LC_from_decomposed_vectorspace_as_dict(vc_dicts: typing.Dict[str, VectorSpace]) -> typing.List[NamedLC]:
    result = []
    for irrepname, vs in vc_dicts.items():
        lcs = vs.get_nonzero_linear_combinations()
        for ilc in range(vs.rank):
            result.append(NamedLC(irrepname, lcs[ilc].get_normalized()))
    return result


def decompose_vectorspace_recursive(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.Dict[str, VectorSpace]:
    subspaces = {}
    group_order = len(group.operations)

    for irrep, characters in group.irreps.items():
        irrep_dimension = group.irrep_dimension[irrep]

        transformed_LCs = []
        for lc in vectorspace.get_nonzero_linear_combinations():
            sum_coefficients = np.zeros_like(lc.coefficients)
            for op, chi in zip(group.operations, characters):
                rotated = lc.symmetry_rotation(op)
                rotated.scale_coefficients(chi * irrep_dimension / group_order)
                sum_coefficients += rotated.coefficients
            transformed_LCs.append(lc.create_new_with_coefficients(sum_coefficients))
        subspaces[irrep] = VectorSpace.from_list_of_linear_combinations(transformed_LCs)

    subduction = subduction_data[group.groupname]

    if subduction["sub"] == "1":
        return subspaces
    else:
        for irrep, subspace in subspaces.items():
            subgroup = group.get_subgroup_from_subgroup_and_seitz_map(
                subduction["sub"], subduction["seitz_map"]
            )
            if irrep in subduction["splittable_rep"]:
                subspaces[irrep] = decompose_vectorspace_recursive(subspace, subgroup)
        return subspaces


def decompose_vectorspace_onelevel(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.Dict[str, VectorSpace]:
    subspaces = {}
    group_order = len(group.operations)

    for irrep, characters in group.irreps.items():
        irrep_dimension = group.irrep_dimension[irrep]

        transformed_LCs = []
        for lc in vectorspace.get_nonzero_linear_combinations():
            sum_coefficients = np.zeros_like(lc.coefficients)
            for op, chi in zip(group.operations, characters):
                rotated = lc.symmetry_rotation(op)
                rotated.scale_coefficients(chi * irrep_dimension / group_order)
                sum_coefficients += rotated.coefficients
                
            transformed_LCs.append(lc.create_new_with_coefficients(sum_coefficients))
        
        subspaces[irrep] = VectorSpace.from_list_of_linear_combinations(transformed_LCs)

    return subspaces


def get_nested_nonzero_vectorspace(space: dict) -> typing.Dict[str,VectorSpace]:
    all_vector_space = {}
    for key, value in space.items():
        if type(value) == dict:
            sub = get_nested_nonzero_vectorspace(value)
            all_vector_space.update({f"{key}->{n}":v for n, v in sub.items()})
        else:
            if value.rank > 0:
                all_vector_space[key] = value
    return all_vector_space