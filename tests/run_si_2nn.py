import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.properties.transport import BandEffectiveMass
from automaticTB.interface import OrbitalPropertyRelationship

prefix = "si_2nn"
result_file = f"{prefix}.result"

cubic_band = [
    "L   0.5   0.5  0.5  G  0.0   0.0  0.0",
    "G   0.0   0.0  0.0  X  0.5   0.0  0.5",
    "X   0.5   0.0  0.5  U  0.625 0.25 0.625",
    "K 0.375 0.375 0.75  G  0.0   0.0  0.0"
]

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
    relationship = OrbitalPropertyRelationship.from_file(result_file)

    interaction_values_2nn = np.array([ 
        -4.81341 * 4, #    1 > Pair: Si-01      3s -> Si-01      3s r = (  0.00,  0.00,  0.00)
        0.01591,      #    2 > Pair: Si-01      3s -> Si-01      3s r = (  2.72,  2.72,  0.00)
        -8.33255,     #    3 > Pair: Si-01      3s -> Si-00      3s r = (  1.36,  1.36,  1.36)
        1.77563 * 4,  #    4 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  0.00,  0.00)
        0.000,        #    5 > Pair: Si-01     3px -> Si-01      3s r = (  0.00,  2.72,  2.72)
        -0.10662,     #    6 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  2.72,  2.72)
        -0.08002,     #    7 > Pair: Si-01     3px -> Si-01      3s r = (  2.72,  2.72,  0.00)
        -0.55067,     #    8 > Pair: Si-01     3px -> Si-01     3py r = (  2.72,  2.72,  0.00)
        2.27784,      #    9 > Pair: Si-01     3px -> Si-01     3pz r = (  2.72,  2.72,  0.00)
        0.00762,      #   10 > Pair: Si-01     3px -> Si-01     3px r = (  2.72,  2.72,  0.00)
        -5.8614,      #   11 > Pair: Si-01     3px -> Si-00      3s r = (  1.36,  1.36,  1.36)
        5.29091,      #   12 > Pair: Si-01     3px -> Si-00     3pz r = (  1.36,  1.36,  1.36)
        1.69916,      #   13 > Pair: Si-01     3px -> Si-00     3px r = (  1.36,  1.36,  1.36)
        6.61342 * 4,  #   14 > Pair: Si-01      4s -> Si-01      4s r = (  0.00,  0.00,  0.00)
        0.000,        #   15 > Pair: Si-01      4s -> Si-01      3s r = (  2.72,  2.72,  0.00)
        1.31699,      #   16 > Pair: Si-01      4s -> Si-01     3pz r = (  2.72,  2.72,  0.00)
        0.080,        #   17 > Pair: Si-01      4s -> Si-01     3px r = (  2.72,  2.72,  0.00)
        0.000,        #   18 > Pair: Si-01      4s -> Si-01      4s r = (  2.72,  2.72,  0.00)
        0.000,        #   19 > Pair: Si-01      4s -> Si-00      3s r = (  1.36,  1.36,  1.36)
        4.8831,       #   20 > Pair: Si-01      4s -> Si-00     3px r = (  1.36,  1.36,  1.36)
        0.000,        #   21 > Pair: Si-01      4s -> Si-00      4s r = (  1.36,  1.36,  1.36)
    ]) / 4.0


    model = relationship.get_ElectronicModel_from_free_parameters(free_Hijs=interaction_values_2nn)
    print(f"# Band structure will be plotted with prefix {prefix}")
    print("")
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band,
        yminymax = (-4,6)
    )

    print(f"Calculate effective mass at CBM")
    mass = BandEffectiveMass(model.tb)
    cb = mass.calc_effective_mass(np.array([0.367, 0.0, 0.367]), ibnd=4)
    print(cb)

solve()
extract()

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
"""
