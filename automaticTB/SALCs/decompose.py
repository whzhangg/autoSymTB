import typing, dataclasses
import numpy as np
from automaticTB.sitesymmetry import subduction_data, SiteSymmetryGroup
from .vectorspace import VectorSpace, get_nonzero_independent_linear_combinations
from .linear_combination import LinearCombination
from automaticTB.parameters import zero_tolerance

@dataclasses.dataclass
class IrrepSymbol:
    symmetry_symbol: str
    main_irrep: str
    main_index: int

    @classmethod
    def from_str(cls, input: str):
        parts = input.split("->")
        mains = parts[0].split("^")
        main_irrep = mains[0]
        main_index = int(mains[1])
        sym_symbol = "->".join([p.split("^")[0] for p in parts])
        return cls(sym_symbol, main_irrep, main_index)

    def __repr__(self) -> str:
        return f"{self.symmetry_symbol} @ {self.main_index}^th {self.main_irrep}"


class NamedLC(typing.NamedTuple):
    name: IrrepSymbol
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
        print(irrepname)
        lcs = vs.get_nonzero_linear_combinations()
        assert vs.rank == len(lcs)
        for lc in lcs:
            print(lc)
            n = IrrepSymbol.from_str(irrepname)
            result.append(NamedLC(n, lc.get_normalized()))
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
                rotated = lc.symmetry_rotate(op)
                rotated.scale_coefficients(chi * irrep_dimension / group_order)
                sum_coefficients += rotated.coefficients
            transformed_LCs.append(lc.create_new_with_coefficients(sum_coefficients))
        
        linearly_independ = get_nonzero_independent_linear_combinations(transformed_LCs)
        if len(linearly_independ) > 0:
            organized = organize_decomposed_lcs(linearly_independ, group, irrep_dimension)
            for i, sublcs in enumerate(organized):
                subspaces[irrep + f"^{i+1}"] = VectorSpace.from_list_of_linear_combinations(sublcs)

    subduction = subduction_data[group.groupname]

    if subduction["sub"] == "1":
        return subspaces
    else:
        subgroup = group.get_subgroup_from_subgroup_and_seitz_map(
                subduction["sub"], subduction["seitz_map"]
            )
        for irrep, subspace in subspaces.items():
            irrep_main = irrep.split("^")[0]
            if irrep_main in subduction["splittable_rep"]:
                subspaces[irrep] = decompose_vectorspace_recursive(subspace, subgroup)
        return subspaces


def organize_decomposed_lcs(decomposedLCs: typing.List[LinearCombination], group: SiteSymmetryGroup, rep_dimension: int) \
-> typing.List[typing.List[LinearCombination]]:
    identified_index = []
    result = []
    for i, lc in enumerate(decomposedLCs):
        if i in identified_index:
            continue
        found = []
        finished = False
        for rot in group.operations:
            rotated = lc.symmetry_rotate(rot)
            for j, jlc in enumerate(decomposedLCs):
                if j in identified_index: continue
                if np.abs(np.dot(rotated.coefficients, jlc.coefficients)) > zero_tolerance:
                    found.append(jlc)
                    identified_index.append(j)
                if len(found) == rep_dimension:
                    finished = True
                    break
            if finished:
                break
        result.append(found)
    return result


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
                rotated = lc.symmetry_rotate(op)
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
