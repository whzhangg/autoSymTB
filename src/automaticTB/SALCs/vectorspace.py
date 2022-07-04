import typing, dataclasses
import numpy as np
from ..structure.sites import Site
from ..structure.nncluster import NearestNeighborCluster
from .linear_combination import LinearCombination, OrbitalsList
from ..parameters import zero_tolerance, LC_coefficients_dtype
from ..tools import remove_zero_vector_from_coefficients, find_linearly_independent_rows, remove_same_direction

def get_nonzero_independent_linear_combinations(inputLCs: typing.List[LinearCombination]) -> typing.List[LinearCombination]:
    coefficient_matrix = np.vstack([
        lc.coefficients for lc in inputLCs
    ])
    result = []
    sites = inputLCs[0].sites
    orbital_list = inputLCs[0].orbital_list
    non_zero = remove_zero_vector_from_coefficients(coefficient_matrix)
    if len(non_zero) > 0:
        non_zero = remove_same_direction(non_zero)
        distinct = find_linearly_independent_rows(non_zero)
            
        for row in distinct:
            result.append(
                LinearCombination(sites, orbital_list, row)
            )
            
    return result

@dataclasses.dataclass
class VectorSpace:
    sites: typing.List[Site]
    orbital_list: OrbitalsList
    coefficients_matrix: np.ndarray  # dtype = complex

    @classmethod
    def from_NNCluster(cls, cluster: NearestNeighborCluster):
        return cls.from_sites_and_orbitals(cluster.baresites, cluster.orbitalslist)

    @classmethod
    def from_sites_and_orbitals(cls, sites: typing.List[Site], orbital_list: OrbitalsList):
        nbasis = orbital_list.num_orb
        coefficients = np.eye(nbasis, dtype=LC_coefficients_dtype)
        return cls(sites, orbital_list, coefficients)

    @classmethod
    def from_list_of_linear_combinations(cls, lcs: typing.List[LinearCombination]):
        sites = lcs[0].sites
        orbital = lcs[0].orbital_list
        coefficients = np.vstack([ lc.coefficients for lc in lcs ])
        return cls(sites, orbital, coefficients)

    @property
    def nbasis(self) -> int:
        return self.orbital_list.num_orb

    @property
    def rank(self):
        return np.linalg.matrix_rank(self.coefficients_matrix, tol = zero_tolerance)

    def __repr__(self) -> str:
        return f"VectorSpace with rank {self.rank} and {len(self.sites)} sites"

    def _remove_linear_dependent(self):
        # remove the redundent linear combination (same direction) or zero 
        non_zero = remove_zero_vector_from_coefficients(self.coefficients_matrix)
        if len(non_zero) > 0:
            distinct = find_linearly_independent_rows(non_zero)
            self.coefficients_matrix = distinct

    def get_nonzero_linear_combinations(self) -> typing.List[LinearCombination]:
        # we do not change the coefficient itself
        result = []
        non_zero = remove_zero_vector_from_coefficients(self.coefficients_matrix)
        if len(non_zero) > 0:
            distinct = find_linearly_independent_rows(non_zero)
            
            for row in distinct:
                result.append(
                    LinearCombination(self.sites, self.orbital_list, row)
                )
            
        return result

