import numpy as np
import typing
from automaticTB.utilities import print_matrix
from automaticTB.parameters import zero_tolerance

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

def find_linearly_independent_rows(coeff_matrix: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/53667174/elimination-the-linear-dependent-columns-of-a-non-square-matrix-in-python
    m,n = coeff_matrix.shape
    assert m <= n, print(f"m={m} n={n}")

    q, r = np.linalg.qr(coeff_matrix.T)
    diagonals = np.diagonal(r)
    indices = np.argwhere(np.abs(diagonals) > zero_tolerance)
    return coeff_matrix[indices.flatten(), :]

if __name__ == "__main__":
    a = np.array([
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0],
    ])

    print(find_linearly_independent_rows(a))