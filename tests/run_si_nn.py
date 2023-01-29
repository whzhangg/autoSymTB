import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship

prefix = "si_nn"
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
        save_filename=result_file
    )


def extract():
    relationship = OrbitalPropertyRelationship.from_file(result_file)

    
    values = np.array([
        -4.81,#    1 > Pair: Si-01      3s -> Si-01      3s r = (  0.00,  0.00,  0.00)
        -8.33/4,#    2 > Pair: Si-01      3s -> Si-00      3s r = (  1.36, -1.36, -1.36)
        1.78,#    3 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  0.00,  0.00)
        -5.86/4,#    4 > Pair: Si-01     3px -> Si-00      3s r = (  1.36, -1.36, -1.36)
        -5.30/4,#    5 > Pair: Si-01     3px -> Si-00     3pz r = (  1.36, -1.36, -1.36)
        1.70/4,#    6 > Pair: Si-01     3px -> Si-00     3px r = (  1.36, -1.36, -1.36)
        6.61,#    7 > Pair: Si-01      4s -> Si-01      4s r = (  0.00,  0.00,  0.00)
        0.000,#    8 > Pair: Si-01      4s -> Si-00      3s r = (  1.36, -1.36, -1.36)
        4.88/4,#    9 > Pair: Si-01      4s -> Si-00     3px r = (  1.36, -1.36, -1.36)
        0.000,#   10 > Pair: Si-01      4s -> Si-00      4s r = (  1.36, -1.36, -1.36)
    ])
    model = relationship.get_ElectronicModel_from_free_parameters(free_Hijs=values)
    print(f"# Band structure will be plotted with prefix {prefix}")
    print("")
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band,
        yminymax = (-4,6)
    )

    #print(f"Calculate effective mass at CBM")
    #mass = BandEffectiveMass(model.tb)
    #cb = mass.calc_effective_mass(np.array([0.367, 0.0, 0.367]), ibnd=4)
    #print(cb)

solve()
#extract()
