import typing

import numpy as np

from automaticTB import parameters as params
from automaticTB.solve import sitesymmetry as ssym

from .vectorspace import VectorSpace, get_nonzero_independent_linear_combinations
from .linear_combination import LinearCombination, IrrepSymbol, NamedLC
from .symmetrygroup import SiteSymmetryGroupwithOrbital, CartesianOrbitalRotation, IrreducibleRep

__all__ = ["decompose_vectorspace_to_namedLC"]


def decompose_vectorspace_to_namedLC(vectorspace: VectorSpace, group: ssym.SiteSymmetryGroup) \
-> typing.List[NamedLC]:
    if group.is_spherical_symmetry:
        return decompose_vectorspace_to_namedLC_sphericalsymmetric(vectorspace)
    else:
        return decompose_vectorspace_to_namedLC_not_sphericalsymmetric(vectorspace, group)


def decompose_vectorspace_to_namedLC_sphericalsymmetric(
    vectorspace: VectorSpace
) -> typing.List[NamedLC]:
    """
    this function treat the special case of spherical symmetric group
    """
    nlms = vectorspace.orbital_list.sh_list
    lcs = vectorspace.get_nonzero_linear_combinations()
    if len(nlms) != len(lcs):
        print("decompose spherical symmetric vectorspace, something wrong")
        raise

    named_lcs = []
    for nlm, lc in zip(nlms, lcs):
        name = f"{nlm.n}-{nlm.l}^1->{nlm.m}"
        named_lcs.append(
            NamedLC(IrrepSymbol.from_str(name), lc)
        )
    
    return named_lcs


def decompose_vectorspace_to_namedLC_not_sphericalsymmetric(
    vectorspace: VectorSpace, group: ssym.SiteSymmetryGroup
) -> typing.List[NamedLC]:
    """
    This function recursively decompose vector space so that the input vector space is decomposed entirely into one-dimensional 
    representations. This method has no knowledge of atoms or basis, it only require that the linearcombination (vectors) know 
    how to rotate themselves given a cartesian rotation
    """
    results = []
    maingroup = SiteSymmetryGroupwithOrbital.from_sitesymmetrygroup_irreps(
        group, vectorspace.orbital_list.irreps_str
    )
    subgroup = maingroup.get_subduction()

    for irrep in maingroup.irreps:
        linearly_independ_lcs = _decompose_onelevel_for_irrep(
            vectorspace.get_nonzero_linear_combinations(), 
            maingroup.operations, 
            irrep
        )

        if len(linearly_independ_lcs) == 0:
            continue # we don't do anything
        
        if irrep.dimension == 1:
            # representation is one dimensional, each linear independent vector is a representation
            for i, lc in enumerate(linearly_independ_lcs):
                symbol = IrrepSymbol.from_str(f"{irrep.name}^{i+1}")
                results.append(
                    NamedLC(symbol, lc.get_normalized())
                )
        else:
            # we need to separate the representations
            if subgroup.groupname == "1":
                print("We have mutlidimension subspace but no subduction")
                raise Exception

            # we first decompose by subgroup and obtain 
            basis_same_symmetry: typing.List[LinearCombination] = None
            for sub_irrep in subgroup.irreps:
                sub_independ_lcs: typing.List[LinearCombination] =\
                    _decompose_onelevel_for_irrep(linearly_independ_lcs, subgroup.operations, sub_irrep)

                if len(sub_independ_lcs) > 0:
                    assert sub_irrep.dimension == 1
                    basis_same_symmetry = sub_independ_lcs # a single basis for each subspace
                    break

            # diagonalize
            assert irrep.dimension * len(basis_same_symmetry) == len(linearly_independ_lcs)

            stacked_coefficients = _get_stacked_coefficients(basis_same_symmetry)
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
                full_basis = _get_orbital(obss, maingroup.operations, irrep.dimension)
                founded_count = 0
                for irrep_sub in subgroup.irreps:
                    further_decomposed = _decompose_onelevel_for_irrep(
                        full_basis, subgroup.operations, irrep_sub
                    )
                    founded_count += len(further_decomposed)
                    assert len(further_decomposed) <= 1
                    if len(further_decomposed) > 0:
                        symbol = IrrepSymbol.from_str(f"{irrep.name}^{i+1}->{irrep_sub.name}")
                        results.append(
                            NamedLC(symbol, further_decomposed[0].get_normalized())
                        )
                if founded_count != irrep.dimension:
                    print(f"For {irrep.name} ({irrep.dimension}), \
                        subduction only found {founded_count} irreps")
        
    return results


def _decompose_onelevel_for_irrep(lcs: typing.List[LinearCombination], operations: typing.List[CartesianOrbitalRotation], irrep: IrreducibleRep) \
-> typing.List[LinearCombination]:

    group_order = len(operations)
    characters = irrep.characters

    transformed_LCs = []
    for lc in lcs:
        sum_coefficients = np.zeros_like(lc.coefficients)
        for op, chi in zip(operations, characters):
            rotated = lc.symmetry_rotate_CartesianOrbital(op)
            # rotated.scale_coefficients(chi * irrep_dimension / group_order)
            #sum_coefficients += rotated.coefficients * (chi * irrep.dimension / group_order)
            # see eq 4.38 in Dresselhaus's book, changed 3/27 20223
            sum_coefficients += rotated.coefficients * (np.conjugate(chi) * irrep.dimension / group_order)

        transformed_LCs.append(lc.create_new_with_coefficients(sum_coefficients))
        
    linearly_independ = get_nonzero_independent_linear_combinations(transformed_LCs)
    assert len(linearly_independ) % irrep.dimension == 0  
    # contain complete irreducible representations
    return linearly_independ


def _get_stacked_coefficients(lcs: typing.List[LinearCombination]) -> np.ndarray:
    """simply stack the coefficients together"""
    return np.vstack([lc.coefficients for lc in lcs])


def _get_orbital(lc: LinearCombination, rotations: typing.List[CartesianOrbitalRotation], number: int) -> typing.List[LinearCombination]:
    matrix = []
    result = []
    rank = 0
    
    for rot in rotations:
        rotated = lc.symmetry_rotate_CartesianOrbital(rot)
        matrix.append(rotated.coefficients)
        new_rank = np.linalg.matrix_rank(np.vstack(matrix), tol = params.ztol)
        if new_rank > rank:
            result.append(rotated)
            rank = new_rank
            if rank == number:
                return result
    print("cannot find all basis ")
    raise 
