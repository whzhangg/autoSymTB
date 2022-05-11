import typing
import numpy as np
from ..linear_combination import LinearCombination, Site, AOLISTS

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
    basis_number = {"s": 1, "p": 3, "d": 5}
    num_defined_aos = len(AOLISTS)
    tolerance = 1e-6
    def __init__(self, sites: typing.List[Site], coefficients_matrix: np.ndarray) -> None:
        assert coefficients_matrix.shape[1] == len(sites) * self.num_defined_aos
        self._sites = sites
        self._vector_coefficients = coefficients_matrix

    @classmethod
    def from_sites(cls, sites: typing.List[Site], starting_basis: str):
        input_dim = len(sites) * cls.num_defined_aos
        starting_basis = starting_basis.split()
        if len(starting_basis) == 0:
            raise "Have to have some basis to start with"
        coefficient_matrix = np.eye(input_dim)
        # put zeros
        if "s" not in starting_basis:
            coefficient_matrix[0::cls.num_defined_aos, :] = 0.0
        if "p" not in starting_basis:
            coefficient_matrix[1::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[2::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[3::cls.num_defined_aos, :] = 0.0
        if "d" not in starting_basis:
            coefficient_matrix[4::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[5::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[6::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[7::cls.num_defined_aos, :] = 0.0
            coefficient_matrix[8::cls.num_defined_aos, :] = 0.0

        non_zero_coefficients = remove_zero_vector_from_coefficients(coefficient_matrix)
        return cls(sites, non_zero_coefficients)

    @classmethod
    def from_list_of_linear_combinations(cls, lcs: typing.List[LinearCombination]):
        sites = lcs[0].sites
        for lc in lcs:
            assert lc.sites == sites

        coefficients = np.array([ lc.coefficients.flatten() for lc in lcs ])
        return cls(sites, coefficients)

    @property
    def rank(self):
        return np.linalg.matrix_rank(self._vector_coefficients, tol = self.tolerance )

    @property
    def sites(self) -> typing.List[Site]:
        return self._sites

    def __repr__(self) -> str:
        return f"VectorSpace with rank {self.rank} and {len(self._sites)} sites"

    def remove_linear_dependent(self):
        # consider a linear combination as vector, the same vector will have 
        non_zero = remove_zero_vector_from_coefficients(self._vector_coefficients)
        if len(non_zero) > 0:
            distinct = remove_same_direction(non_zero)
            self._vector_coefficients = distinct

    def get_nonzero_linear_combinations(self) -> typing.List[LinearCombination]:
        result = []
        for row in self._vector_coefficients:
            if not np.isclose(np.linalg.norm(row), 0.0):
                result.append(
                    LinearCombination(self._sites, row.reshape((len(self._sites), self.num_defined_aos)))
                )
        return result