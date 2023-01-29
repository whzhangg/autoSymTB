import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship

prefix = "perovskite"

def solve():
    current_folder = os.path.split(__file__)[0]
    pbte_file = os.path.join(current_folder, "..", "resources", "perovskite.cif")
    relationship = solve_interaction(
        structure=pbte_file,
        orbitals_dict={"Pb":"6s6p", "Cl": "3s3p"},
        rcut = 3.0,
        save_filename=f"{prefix}.result"
    )

def extract():
    cubic_band = [
        "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
        "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
        "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
        "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
        "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
    ]

    interaction_values = np.array([
        -9.21,    #    1 > Pair: Pb-00      6s -> Pb-00      6s r = (  0.00,  0.00,  0.00)
         2.39,    #    2 > Pair: Pb-00     6px -> Pb-00     6px r = (  0.00,  0.00,  0.00)
       -13.21,    #    3 > Pair: Cl-03      3s -> Cl-03      3s r = (  0.00,  0.00,  0.00)
        -1.10,    #    4 > Pair: Cl-03      3s -> Pb-00      6s r = (  0.00,  0.00,  2.84)
        -0.60,    #    5 > Pair: Cl-03      3s -> Pb-00     6pz r = (  0.00,  0.00,  2.84)
        -1.25,    #    6 > Pair: Cl-03     3pz -> Pb-00      6s r = (  0.00,  0.00,  2.84)
        -3.50,    #    7 > Pair: Cl-03     3pz -> Pb-00     6pz r = (  0.00,  0.00,  2.84)
        -2.16,    #    8 > Pair: Cl-03     3px -> Cl-03     3px r = (  0.00,  0.00,  0.00)
         0.80,    #    9 > Pair: Cl-03     3px -> Pb-00     6px r = (  0.00,  0.00,  2.84)
    ])

    relationship = OrbitalPropertyRelationship.from_file(f"{prefix}.result")
    model = relationship.get_ElectronicModel_from_free_parameters(free_Hijs=interaction_values)
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band
    )

extract()