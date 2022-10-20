from automaticTB.tools.mathLA import *
import numpy as np

def test_find_linearly_independence_rows():
    """
    We create a 10 * 20 array with random numbers as linearly independent inputs, 
    Since the numbers are given in random, linear independence is not guaranteed, 
    but is highly likely. 
    We use linear transformation to create the rest of the rows which would be linear dependent,
    they are stacked and pass through the function. 
    We also added a row of zero, which is removed automatically
    """
    ncol = 20
    nrow = 9
    random_input = np.random.rand(nrow, ncol) - 0.5 
    transform = np.random.rand(nrow, nrow) - 0.5
    dependent_rows = np.einsum("ij, jk -> ik", transform, random_input)
    combined_input = np.vstack([
        random_input, 
        dependent_rows,
        np.zeros(ncol)
    ])
    assert len(combined_input) == nrow * 2 + 1
    independent_inputs = find_linearly_independent_rows(combined_input)
    assert len(independent_inputs) == nrow


def test_same_direction_and_linear_independence():
    """
    this test examines that remove_parallel_vectors() works as expected, 
    as well as find_linearly_independent_rows()
    """ 
    ncol = 20
    nrow = 10
    random_input = np.random.rand(nrow, ncol) - 0.5 
    transform = np.diag(np.sign(np.random.rand(nrow)))
    dependent_rows = np.einsum("ij, jk -> ik", transform, random_input)
    combined_input = np.vstack([
            random_input, 
            dependent_rows,
        ])
    result1 = remove_parallel_vectors(combined_input)
    result2 = find_linearly_independent_rows(combined_input)
    assert result1.shape == result2.shape
    assert np.allclose(result1, result2)

