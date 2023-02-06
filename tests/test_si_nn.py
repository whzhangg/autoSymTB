import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from compare_values import compare_stored_values, StoredInteractions

prefix = "si_nn"
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
        save_filename=result_file
    )


def extract():
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    print(f"# Band structure will be plotted with prefix {prefix}")
    print("")
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = cubic_band,
        yminymax = (-4,6)
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
        kpaths_str = cubic_band,
        yminymax = (-4,6)
    )
    os.remove(result_file)

if __name__ == "__main__":
    solve()
    extract()

"""
values = np.array([
    -4.81,  #    1 > Pair: Si-01      3s -> Si-01      3s r = (  0.00,  0.00,  0.00)
    -8.33/4,#    2 > Pair: Si-01      3s -> Si-00      3s r = (  1.36, -1.36, -1.36)
    1.78,   #    3 > Pair: Si-01     3px -> Si-01     3px r = (  0.00,  0.00,  0.00)
    -5.86/4,#    4 > Pair: Si-01     3px -> Si-00      3s r = (  1.36, -1.36, -1.36)
    -5.30/4,#    5 > Pair: Si-01     3px -> Si-00     3pz r = (  1.36, -1.36, -1.36)
    1.70/4, #    6 > Pair: Si-01     3px -> Si-00     3px r = (  1.36, -1.36, -1.36)
    6.61,   #    7 > Pair: Si-01      4s -> Si-01      4s r = (  0.00,  0.00,  0.00)
    0.000,  #    8 > Pair: Si-01      4s -> Si-00      3s r = (  1.36, -1.36, -1.36)
    4.88/4, #    9 > Pair: Si-01      4s -> Si-00     3px r = (  1.36, -1.36, -1.36)
    0.000,  #   10 > Pair: Si-01      4s -> Si-00      4s r = (  1.36, -1.36, -1.36)
])
"""