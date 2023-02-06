import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from compare_values import compare_stored_values, StoredInteractions

prefix = "perovskite"
result_file = f"{prefix}.result"

cubic_band = [
    "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
    "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
    "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
    "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
    "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
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
    pbte_file = os.path.join(current_folder, "..", "resources", "perovskite.cif")
    relationship = solve_interaction(
        structure=pbte_file,
        orbitals_dict={"Pb":"6s6p", "Cl": "3s3p"},
        rcut = 3.0,
        save_filename=result_file
    )


def extract():
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(free_Hijs=retrived_values)
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band
    )


def test_solved_values():
    solve()
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    compare_stored_values(prefix, model.tb, cubic_band)
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band
    )
    os.remove(result_file)

if __name__ == "__main__":
    extract()

"""
interaction_values = np.array([
    -9.21,    #    1 > Pair: Pb-00      6s -> Pb-00      6s r = (  0.00,  0.00,  0.00)
     2.39,    #    2 > Pair: Pb-00     6px -> Pb-00     6px r = (  0.00,  0.00,  0.00)
    -13.21,   #    3 > Pair: Cl-03      3s -> Cl-03      3s r = (  0.00,  0.00,  0.00)
    -1.10,    #    4 > Pair: Cl-03      3s -> Pb-00      6s r = (  0.00,  0.00,  2.84)
    -0.60,    #    5 > Pair: Cl-03      3s -> Pb-00     6pz r = (  0.00,  0.00,  2.84)
    -1.25,    #    6 > Pair: Cl-03     3pz -> Pb-00      6s r = (  0.00,  0.00,  2.84)
    -3.50,    #    7 > Pair: Cl-03     3pz -> Pb-00     6pz r = (  0.00,  0.00,  2.84)
    -2.16,    #    8 > Pair: Cl-03     3px -> Cl-03     3px r = (  0.00,  0.00,  0.00)
     0.80,    #    9 > Pair: Cl-03     3px -> Pb-00     6px r = (  0.00,  0.00,  2.84)
])
"""