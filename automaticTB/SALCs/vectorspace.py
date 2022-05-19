import typing, dataclasses
import numpy as np
from ..structure.sites import Site
from ..structure.nncluster import NearestNeighborCluster
from .linear_combination import LinearCombination, OrbitalsList
from ..parameters import zero_tolerance


@dataclasses.dataclass
class VectorSpace:
    sites: typing.List[Site]
    orbital_list: OrbitalsList
    coefficients_matrix: np.ndarray

    @classmethod
    def from_NNCluster(cls, cluster: NearestNeighborCluster):
        return cls.from_sites_and_orbitals(cluster.baresites, cluster.orbitalslist)

    @classmethod
    def from_sites_and_orbitals(cls, sites: typing.List[Site], orbital_list: OrbitalsList):
        nbasis = orbital_list.num_orb
        coefficients = np.eye(nbasis, dtype=float)
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
        non_zero = _remove_zero_vector_from_coefficients(self.coefficients_matrix)
        if len(non_zero) > 0:
            distinct = _remove_same_direction(non_zero)
            self.coefficients_matrix = distinct

    def get_nonzero_linear_combinations(self) -> typing.List[LinearCombination]:
        self._remove_linear_dependent()
        result = []
        for row in self.coefficients_matrix:
            result.append(
                LinearCombination(self.sites, self.orbital_list, row)
            )
        return result


def _remove_zero_vector_from_coefficients(coeff_matrix: np.ndarray) -> np.ndarray:
    is_zero = np.isclose(np.linalg.norm(coeff_matrix, axis = 1), 0.0)
    not_zero = np.invert(is_zero)
    return coeff_matrix[not_zero, :]


def _remove_same_direction(vectors: np.ndarray) -> np.ndarray:
    distinct_direction = [vectors[0]]
    for vector in vectors:
        is_close = False
        for compare in distinct_direction:
            cos = np.dot(vector, compare) / np.linalg.norm(vector) / np.linalg.norm(compare)
            if np.isclose(np.abs(cos), 1.0):
                is_close = True
                break
        if not is_close:
            distinct_direction.append(vector)
    return np.array(distinct_direction)
