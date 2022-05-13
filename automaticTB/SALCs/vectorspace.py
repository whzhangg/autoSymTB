import typing
import numpy as np
from ..linear_combination import LinearCombination, Site, Orbitals

def remove_zero_vector_from_coefficients(coeff_matrix: np.ndarray) -> np.ndarray:
    is_zero = np.isclose(np.linalg.norm(coeff_matrix, axis = 1), 0.0)
    not_zero = np.invert(is_zero)
    return coeff_matrix[not_zero, :]

def remove_same_direction(vectors: np.ndarray) -> np.ndarray:
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

class VectorSpace:
    """
    it will be a subspace spaned by all the given basis vector, as well as the group operations,
    each linear combination serve as an initial basis function
    """
    _rank_tolerance = 1e-6

    def __init__(self, sites: typing.List[Site], orbital: Orbitals, coefficients_matrix: np.ndarray) -> None:
        self._orbital = orbital
        assert coefficients_matrix.shape[1] == len(sites) * self._orbital.num_orb

        self._sites = sites
        self._vector_coefficients = coefficients_matrix

    @classmethod
    def from_sites(cls, sites: typing.List[Site], starting_basis: str):
        orbital = Orbitals(starting_basis)
        coefficient_matrix = np.eye(
            len(sites) * orbital.num_orb, dtype=float
        )
        return cls(sites, orbital, coefficient_matrix)

    @classmethod
    def from_list_of_linear_combinations(cls, lcs: typing.List[LinearCombination]):
        sites = lcs[0].sites
        orbital = lcs[0].orbitals
        for lc in lcs:
            assert lc.sites == sites
            assert lc.orbitals == orbital

        coefficients = np.array([ lc.coefficients.flatten() for lc in lcs ])
        return cls(sites, orbital, coefficients)

    @property
    def orbitals(self) -> Orbitals:
        return self._orbital

    @property
    def rank(self):
        return np.linalg.matrix_rank(self._vector_coefficients, tol = self._rank_tolerance )

    @property
    def sites(self) -> typing.List[Site]:
        return self._sites

    def __repr__(self) -> str:
        return f"VectorSpace with rank {self.rank} and {len(self._sites)} sites"

    def _remove_linear_dependent(self):
        # remove the redundent linear combination (same direction) or zero 
        non_zero = remove_zero_vector_from_coefficients(self._vector_coefficients)
        if len(non_zero) > 0:
            distinct = remove_same_direction(non_zero)
            self._vector_coefficients = distinct

    def get_nonzero_linear_combinations(self) -> typing.List[LinearCombination]:
        self._remove_linear_dependent()
        result = []
        shape = (len(self.sites), self._orbital.num_orb)
        for row in self._vector_coefficients:
            result.append(
                LinearCombination(self._sites, self._orbital, row.reshape(shape))
            )
        return result