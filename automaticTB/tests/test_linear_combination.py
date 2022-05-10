from automaticTB.linear_combination import LinearCombination, Site
import numpy as np

sample_cluster_CH4 = [
    Site( 6, np.array([ 0, 0, 0]) ),
    Site( 1, np.array([ 1, 1, 1]) ),
    Site( 1, np.array([-1,-1, 1]) ),
    Site( 1, np.array([ 1,-1,-1]) ),
    Site( 1, np.array([-1, 1,-1]) )
]

coefficients = np.random.rand(5,9) - 0.5

def test_return_normalized_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, coefficients)
    normalized_lc = lc.normalized()
    assert normalized_lc.is_normalized

def test_normalized_contruction_of_linear_combination():
    normalized_lc = LinearCombination(sample_cluster_CH4, coefficients, normalize=True)
    assert normalized_lc.is_normalized

def print_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, coefficients)
    print(lc)
    print(lc.pretty_str)