from automaticTB.linear_combination import LinearCombination, Site, Orbitals
import numpy as np

sample_cluster_CH4 = [
    Site( 6, np.array([ 0, 0, 0]) ),
    Site( 1, np.array([ 1, 1, 1]) ),
    Site( 1, np.array([-1,-1, 1]) ),
    Site( 1, np.array([ 1,-1,-1]) ),
    Site( 1, np.array([-1, 1,-1]) )
]

coefficients = np.random.rand(5,9) - 0.5

def test_Orbitals():
    orbitals = Orbitals("s p d")
    assert orbitals.num_orb == 9
    assert orbitals.irreps == '1x0e + 1x1o + 1x2e'
    assert orbitals.slice_dict == {
        "s": slice(0,1), "p": slice(1,4), "d": slice(4,9)
    }


def test_Orbitals_combined():
    orbitals = Orbitals("s d")
    assert orbitals.num_orb == 6
    assert orbitals.irreps == '1x0e + 1x2e'
    assert orbitals.slice_dict == {
        "s": slice(0,1), "d": slice(1,6)
    }


def test_slicing_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, coefficients)
    for orb in ["s", "p", "d"]:
        new_lc = lc.get_orbital_subspace(orb)
        assert new_lc.coefficients.shape[1] == {"s": 1, "p": 3, "d": 5}[orb]

def test_initialize_linear_combination_of_p():
    lc = LinearCombination(sample_cluster_CH4, "p", coefficients[:,1:4])

def test_initialize_linear_combination_of_d():
    lc = LinearCombination(sample_cluster_CH4, "d", coefficients[:,4:9])

def test_initialize_linear_combination_of_sd():
    lc = LinearCombination(sample_cluster_CH4, "s p", coefficients[:,0:4])

def test_return_normalized_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, "s p d", coefficients)
    normalized_lc = lc.normalized()
    assert normalized_lc.is_normalized

def test_normalized_contruction_of_linear_combination():
    normalized_lc = LinearCombination(sample_cluster_CH4, "s p d", coefficients, normalize=True)
    assert normalized_lc.is_normalized

def test_print_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, "s p d", coefficients)
    print(lc)
    print(lc.pretty_str)