import typing, dataclasses
import numpy as np
from automaticTB.sitesymmetry import subduction_data, SiteSymmetryGroup
from .vectorspace import VectorSpace, get_nonzero_independent_linear_combinations
from .linear_combination import LinearCombination
from automaticTB.parameters import zero_tolerance
from .named_lc import IrrepSymbol, NamedLC
from .symmetrygroup import SiteSymmetryGroupwithOrbital, CartesianOrbitalRotation, IrreducibleRep

# recursive
def decompose_vectorspace_to_namedLC(vectorspace: VectorSpace, group: SiteSymmetryGroup) \
-> typing.List[NamedLC]:
    results = []
    maingroup = SiteSymmetryGroupwithOrbital.from_sitesymmetrygroup_irreps(group, vectorspace.orbital_list.irreps_str)
    subgroup = maingroup.get_subduction()

    for irrep in maingroup.irreps:
        linearly_independ_lcs = decompose_onelevel_for_irrep(vectorspace.get_nonzero_linear_combinations(), maingroup.operations, irrep)

        if len(linearly_independ_lcs) == 0:
            continue # we don't do anything
        
        if irrep.dimension == 1:
            for i, lc in enumerate(linearly_independ_lcs):
                symbol = IrrepSymbol.from_str(f"{irrep.name}^{i+1}")
                results.append(
                    NamedLC(symbol, lc.get_normalized())
                )
        else:
            if subgroup.groupname == "1":
                print("We have mutlidimension subspace but no subduction")
                raise Exception

            # we first decompose 
            basis_same_symmetry: typing.List[LinearCombination] = None
            for sub_irrep in subgroup.irreps:
                sub_independ_lcs = decompose_onelevel_for_irrep(linearly_independ_lcs, subgroup.operations, sub_irrep)

                if len(sub_independ_lcs) > 0:
                    assert sub_irrep.dimension == 1
                    basis_same_symmetry = sub_independ_lcs
                    break

            # diagonalize
            assert irrep.dimension * len(basis_same_symmetry) == len(linearly_independ_lcs)

            stacked_coefficients = get_stacked_coefficients(basis_same_symmetry)
            identity = np.matmul(stacked_coefficients, stacked_coefficients.T)
            w, v = np.linalg.eig(identity)
            starting_basis = np.linalg.inv(v) @ stacked_coefficients
            
            orthogonal_basis_same_symmetry = []
            for sb in starting_basis:
                orthogonal_basis_same_symmetry.append(
                    LinearCombination(vectorspace.sites, vectorspace.orbital_list, sb)
                )
            # orthogonal basis 

            for i, obss in enumerate(orthogonal_basis_same_symmetry):
                full_basis = get_orbital(obss, maingroup.operations, irrep.dimension)
                for irrep_sub in subgroup.irreps:
                    further_decomposed = decompose_onelevel_for_irrep(full_basis, subgroup.operations, irrep_sub)
                    assert len(further_decomposed) <= 1
                    if len(further_decomposed) > 0:
                        symbol = IrrepSymbol.from_str(f"{irrep.name}^{i+1}->{irrep_sub.name}")
                        results.append(
                            NamedLC(symbol, further_decomposed[0].get_normalized())
                        )
        
    return results


def decompose_onelevel_for_irrep(lcs: typing.List[LinearCombination], operations: typing.List[CartesianOrbitalRotation], irrep: IrreducibleRep) \
-> typing.Dict[str, typing.List[LinearCombination]]:

    group_order = len(operations)
    characters = irrep.characters

    transformed_LCs = []
    for lc in lcs:
        sum_coefficients = np.zeros_like(lc.coefficients)
        for op, chi in zip(operations, characters):
            rotated = lc.symmetry_rotate_CartesianOrbital(op)
            # rotated.scale_coefficients(chi * irrep_dimension / group_order)
            sum_coefficients += rotated.coefficients * (chi * irrep.dimension / group_order)

        transformed_LCs.append(lc.create_new_with_coefficients(sum_coefficients))
        
    linearly_independ = get_nonzero_independent_linear_combinations(transformed_LCs)
    assert len(linearly_independ) % irrep.dimension == 0  # contain complete irreducible representations
    
    return linearly_independ


def get_stacked_coefficients(lcs: typing.List[LinearCombination]) -> np.ndarray:
    return np.vstack([lc.coefficients for lc in lcs])


def get_orbital(lc: LinearCombination, rotations: typing.List[CartesianOrbitalRotation], number: int) -> typing.List[LinearCombination]:
    matrix = []
    result = []
    rank = 0
    
    for rot in rotations:
        rotated = lc.symmetry_rotate_CartesianOrbital(rot)
        matrix.append(rotated.coefficients)
        new_rank = np.linalg.matrix_rank(np.vstack(matrix))
        if new_rank > rank:
            result.append(rotated)
            rank = new_rank
            if rank == number:
                return result
    print("cannot find all basis ")
    raise 