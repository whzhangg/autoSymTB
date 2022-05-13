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

def test_Orbitals_properties():
    orbitals = Orbitals("s d")
    assert orbitals.num_orb == 6
    assert orbitals.irreps_str == '1x0e + 1x2e'
    assert orbitals.slice_dict == {
        "s": slice(0,1), "d": slice(1,6)
    }


def test_linear_combination_normalization():
    lc = LinearCombination(sample_cluster_CH4, Orbitals("s p d"), coefficients)
    lc_normalized_initiated = LinearCombination(sample_cluster_CH4, Orbitals("s p d"), coefficients, normalize=True)
    
    assert lc.normalized().is_normalized
    assert lc_normalized_initiated.is_normalized


def test_slicing_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, Orbitals("s p d"), coefficients)
    for orb in ["s", "p", "d"]:
        new_lc = lc.get_orbital_subspace(Orbitals(orb))
        assert new_lc.coefficients.shape[1] == {"s": 1, "p": 3, "d": 5}[orb]


def test_print_linear_combination():
    lc = LinearCombination(sample_cluster_CH4, Orbitals("s p d"), coefficients)
    print(lc)
