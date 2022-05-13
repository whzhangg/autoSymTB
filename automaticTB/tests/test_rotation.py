from automaticTB.linear_combination import LinearCombination, Site
from automaticTB.rotation import rotate_linear_combination_from_matrix
from automaticTB.utilities import random_rotation_matrix
import numpy as np

sample_cluster_CH4 = [
    Site( 6, np.array([ 0, 0, 0]) ),
    Site( 1, np.array([ 1, 1, 1]) ),
    Site( 1, np.array([-1,-1, 1]) ),
    Site( 1, np.array([ 1,-1,-1]) ),
    Site( 1, np.array([-1, 1,-1]) )
]

coefficients = np.random.rand(5,9) - 0.5

def test_initialize_linear_combination_of_s():
    lc = LinearCombination(sample_cluster_CH4, "s", coefficients[:,0:1])
    rot = random_rotation_matrix()
    rotate_linear_combination_from_matrix(lc, rot)

def test_initialize_linear_combination_of_p():
    lc = LinearCombination(sample_cluster_CH4, "p", coefficients[:,1:4])
    rot = random_rotation_matrix()
    rotate_linear_combination_from_matrix(lc, rot)

def test_initialize_linear_combination_of_d():
    lc = LinearCombination(sample_cluster_CH4, "d", coefficients[:,4:9])
    rot = random_rotation_matrix()
    rotate_linear_combination_from_matrix(lc, rot)

def test_initialize_linear_combination_of_sd():
    lc = LinearCombination(sample_cluster_CH4, "s p", coefficients[:,0:4])
    rot = random_rotation_matrix()
    rotate_linear_combination_from_matrix(lc, rot)
