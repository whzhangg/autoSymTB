import typing
import dataclasses

import numpy as np
from automaticTB import tools
from automaticTB import parameters as params
from automaticTB.solve import structure
from automaticTB.solve import atomic_orbitals
from .linear_combination import LinearCombination


def get_nonzero_independent_linear_combinations(inputLCs: typing.List[LinearCombination]) \
-> typing.List[LinearCombination]:
    """
    this function removes the linearly dependent vector in a list of input linear combinations 
    """
    coefficient_matrix = np.vstack([
        lc.coefficients for lc in inputLCs
    ])
    sites = inputLCs[0].sites
    orbital_list = inputLCs[0].orbital_list

    result = []
    distinct = tools.get_distinct_nonzero_vector_from_coefficients(coefficient_matrix)
    for row in distinct:
        result.append(
            LinearCombination(sites, orbital_list, row)
        )
    return result


@dataclasses.dataclass
class VectorSpace:
    sites: typing.List[structure.LocalSite]
    orbital_list: atomic_orbitals.OrbitalsList
    coefficients_matrix: np.ndarray  # dtype = complex

    @classmethod
    def from_sites_and_orbitals(cls, sites: typing.List[structure.LocalSite], orbital_list: atomic_orbitals.OrbitalsList):
        nbasis = orbital_list.num_orb
        coefficients = np.eye(nbasis, dtype=params.COMPLEX_TYPE)
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
        return np.linalg.matrix_rank(self.coefficients_matrix, tol = params.ztol)


    def __repr__(self) -> str:
        return f"VectorSpace with rank {self.rank} and {len(self.sites)} sites"


    def get_nonzero_linear_combinations(self) -> typing.List[LinearCombination]:
        """
        return a list on non-zero linear combinations in this vector space
        """
        result = []
        distinct = tools.get_distinct_nonzero_vector_from_coefficients(self.coefficients_matrix)
        for row in distinct:
            result.append(
                LinearCombination(self.sites, self.orbital_list, row)
            )
        return result

