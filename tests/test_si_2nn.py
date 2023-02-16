import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.properties.transport import BandEffectiveMass, ParabolicBandFitting, TaylorBandFitting
from automaticTB.interface import OrbitalPropertyRelationship
from compare_values import compare_stored_values, StoredInteractions

prefix = "si_2nn"
result_file = f"{prefix}.result"

cubic_band = [
    "L   0.5   0.5  0.5  G  0.0   0.0  0.0",
    "G   0.0   0.0  0.0  X  0.5   0.0  0.5",
    "X   0.5   0.0  0.5  U  0.625 0.25 0.625",
    "K 0.375 0.375 0.75  G  0.0   0.0  0.0"
]
def get_needed_values(aopairs) -> np.ndarray:
    from compare_values import StoredInteractions
    stored = StoredInteractions.from_system_name(prefix)
    retrived_values = []
    for aopair in aopairs:
        retrived_values.append(stored.find_value(aopair))
    retrived_values = np.array(retrived_values)
    return retrived_values


def solve():
    current_folder = os.path.split(__file__)[0]
    pbte_file = os.path.join(current_folder, "..", "resources", "Si_ICSD51688.cif")

    solve_interaction(
        structure=pbte_file,
        orbitals_dict={"Si":"3s3p4s"},
        rcut=3.9,
        save_filename=result_file
    )


def extract():
    """plot the band structure and so on"""
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    #np.save("si_2nn_homogeneous.npy", relationship.homogeneous_equation)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    print(f"# Band structure will be plotted with prefix {prefix}")
    print("")
    banddata = model.get_bandstructure(
        prefix, cubic_band, quality = 1, make_folder=False
    )
    banddata.plot_data(f"{prefix}.pdf", (-4,6))
    dos = model.get_dos(prefix, emin=-4.0, emax=6.0, gridsize=40, xdensity=200)
    dos.plot_dos("si_2nn.dos.pdf")
    return
    print(f"Calculate effective mass at CBM (parabolic mode)")
    for k in np.linspace((0.3,0.0,0.3),(0.5,0.0,0.5),15):
        parabolic_mass: ParabolicBandFitting = model.get_effectivemass(
            k, ibnd = 4, mode="parabolic")
        print("selected k = {:>8.4f}{:>8.4f}{:>8.4f}".format(*k), end = " ")
        print("k(emin) = {:>8.4f}{:>8.4f}{:>8.4f}".format(*parabolic_mass.kmin), end = " ")
        print("EM = {:>8.4f}{:>8.4f}{:>8.4f}".format(*(1 / np.diag(parabolic_mass.inv_mass))))
    

def test_solved_values():
    solve()
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    compare_stored_values(prefix, model.tb, cubic_band)
    
    banddata = model.get_bandstructure(
        prefix, cubic_band, quality = 1, make_folder=False
    )
    banddata.plot_data(f"{prefix}.pdf", (-4,6))
    os.remove(result_file)

    dos = model.get_dos(prefix, emin=-4.0, emax=6.0)
    dos.plot_dos("si_2nn.dos.pdf")
    assert np.isclose(dos.nsum(0.5), 8.0)


if __name__ == "__main__":
    solve()
    #debug()
    #test_solved_values()
    extract()
    pass
"""
Effective mass results:
Calculate effective mass at CBM
{
    'k0_frac': array([0.36465333, 0.        , 0.36465333]), 
    'e0': 1.4253161101088576, 
    'inv_mass_tensor': array([[5.20770477e+00, 4.72332579e-06, 3.31865647e-05],
                              [4.72332579e-06, 1.08769291e+00, 3.51139798e-05],
                              [3.31865647e-05, 3.51139798e-05, 5.20771183e+00]]), 
    'eigen_mass': array([0.19202317, 0.91937715, 0.19202291])
}


interaction_values_2nn = np.array([
    -4.81341 * 4, #    1 > Pair: Si-01      3s -> Si-01      3s r = (  0.00,  0.00,  0.00) -
    0.01591,      #    2 > Pair: Si-01      3s -> Si-01      3s r = (  2.72,  2.72,  0.00) -
    -8.33255,     #    3 > Pair: Si-01      3s -> Si-00      3s r = (  1.36,  1.36,  1.36) -
    1.77563 * 4,  #    4 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  0.00,  0.00) -
    1.317,        #    5 > Pair: Si-01     3px -> Si-01      3s r = (  0.00,  2.72,  2.72)
    -0.10662,     #    6 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  2.72,  2.72) -
    -0.08002,     #    7 > Pair: Si-01     3px -> Si-01      3s r = (  2.72,  2.72,  0.00) -
     0.55067,     #    8 > Pair: Si-01     3px -> Si-01     3py r = (  2.72,  2.72,  0.00) -
    2.27784,      #    9 > Pair: Si-01     3px -> Si-01     3pz r = (  2.72,  2.72,  0.00)
    0.00762,      #   10 > Pair: Si-01     3px -> Si-01     3px r = (  2.72,  2.72,  0.00) -
    -5.8614,      #   11 > Pair: Si-01     3px -> Si-00      3s r = (  1.36,  1.36,  1.36) -
    5.29091,      #   12 > Pair: Si-01     3px -> Si-00     3pz r = (  1.36,  1.36,  1.36) -
    1.69916,      #   13 > Pair: Si-01     3px -> Si-00     3px r = (  1.36,  1.36,  1.36) -
    5.61342 * 4,  #   14 > Pair: Si-01      4s -> Si-01      4s r = (  0.00,  0.00,  0.00) -
    0.000,        #   15 > Pair: Si-01      4s -> Si-01      3s r = (  2.72,  2.72,  0.00) -
    0.50103,      #   16 > Pair: Si-01      4s -> Si-01     3pz r = (  2.72,  2.72,  0.00)
    -0.0058,      #   17 > Pair: Si-01      4s -> Si-01     3px r = (  2.72,  2.72,  0.00) -
    0.000,        #   18 > Pair: Si-01      4s -> Si-01      4s r = (  2.72,  2.72,  0.00) -
    0.000,        #   19 > Pair: Si-01      4s -> Si-00      3s r = (  1.36,  1.36,  1.36) -
    4.8831,       #   20 > Pair: Si-01      4s -> Si-00     3px r = (  1.36,  1.36,  1.36) -
    0.000,        #   21 > Pair: Si-01      4s -> Si-00      4s r = (  1.36,  1.36,  1.36) -
]) / 4.0

interaction_values_2nn_cp = np.array([
    0.00762,     #    1 > Pair: Si-00     3pz -> Si-00     3pz r = ( -2.72,  0.00, -2.72) -
    -4.81341 * 4, #    2 > Pair: Si-01      3s -> Si-01      3s r = (  0.00,  0.00,  0.00) -
    0.01591,      #    3 > Pair: Si-01      3s -> Si-01      3s r = ( -2.72,  0.00, -2.72) -
    -5.8614,      #    4 > Pair: Si-01      3s -> Si-00     3px r = ( -1.36,  1.36, -1.36) -
    -1.317,        #    5 > Pair: Si-01      3s -> Si-01     3px r = (  0.00,  2.72, -2.72) ?
    -8.33255,     #    6 > Pair: Si-01      3s -> Si-00      3s r = (  1.36,  1.36,  1.36) -
    1.77563 * 4,  #    7 > Pair: Si-01     3py -> Si-01     3py r = (  0.00,  0.00,  0.00) -
    4.8831,       #    8 > Pair: Si-01     3py -> Si-00      4s r = ( -1.36,  1.36, -1.36) -
     0.08002,     #    9 > Pair: Si-01     3py -> Si-01      3s r = (  2.72, -2.72,  0.00) -
    -0.0058,      #   10 > Pair: Si-01     3py -> Si-01      4s r = (  2.72, -2.72,  0.00) -
    1.69916,      #   11 > Pair: Si-01     3py -> Si-00     3py r = (  1.36,  1.36,  1.36) -
    -0.10662,     #   12 > Pair: Si-01     3pz -> Si-01     3pz r = ( -2.72, -2.72,  0.00) -
    -0.55067,     #   13 > Pair: Si-01     3pz -> Si-01     3py r = (  0.00,  2.72, -2.72) -
    5.29091,      #   14 > Pair: Si-01     3pz -> Si-00     3px r = (  1.36,  1.36,  1.36) -
    -2.27784,      #   15 > Pair: Si-01     3px -> Si-01     3py r = (  0.00, -2.72,  2.72) ?
    5.61342 * 4,  #   16 > Pair: Si-01      4s -> Si-01      4s r = (  0.00,  0.00,  0.00) -
    0.50103,      #   17 > Pair: Si-01      4s -> Si-01     3px r = (  0.00,  2.72, -2.72) ?
    0.000,        #   18 > Pair: Si-01      4s -> Si-01      3s r = (  2.72,  2.72,  0.00) -
    0.000,        #   19 > Pair: Si-01      4s -> Si-01      4s r = (  2.72,  2.72,  0.00) -
    0.000,        #   20 > Pair: Si-01      4s -> Si-00      3s r = (  1.36,  1.36,  1.36) - 
    0.000,        #   21 > Pair: Si-01      4s -> Si-00      4s r = (  1.36,  1.36,  1.36) -
]) / 4.0
"""
