import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from compare_values import compare_stored_values

prefix = "pbte"
result_file = f"{prefix}.result"

pbte_path = [
    'G 0.0000 0.0000 0.0000 X 0.5000 0.0000 0.5000', 
    'X 0.5000 0.0000 0.5000 W 0.5000 0.2500 0.7500', 
    'W 0.5000 0.2500 0.7500 L 0.5000 0.5000 0.5000', 
    'L 0.5000 0.5000 0.5000 G 0.0000 0.0000 0.0000', 
    'G 0.0000 0.0000 0.0000 K 0.3750 0.3750 0.7500'
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
    pbte_file = os.path.join(current_folder, "..", "resources", "PbTe_ICSD38295.cif")
    solve_interaction(
        structure=pbte_file,
        orbitals_dict={"Pb":"6s6p", "Te": "5s5p"},
        rcut = 4.7,
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
        kpaths_str = pbte_path,
        yminymax=(8,12)
    )


def test_solved_values():
    solve()
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    compare_stored_values(prefix, model.tb, pbte_path)
    bandresult = model.get_bandstructure(prefix, pbte_path, make_folder=False)
    bandresult.plot_data(f"{prefix}.pdf", yminymax=(8,12))
    os.remove(result_file)


if __name__ == "__main__":
    solve()
    extract()

"""
R = np.sqrt(2)
values = np.array([
        2.6, #    1 > Pair: Pb-00      6s -> Pb-00      6s r = (  0.00,  0.00,  0.00)
        -0.08,#    2 > Pair: Pb-00      6s -> Pb-00      6s r = (  3.23,  3.23,  0.00)
        11.0,#    3 > Pair: Pb-00     6px -> Pb-00     6px r = (  0.00,  0.00,  0.00)
        -0.04,#    4 > Pair: Pb-00     6px -> Pb-00     6px r = (  0.00,  3.23,  3.23)
        -0.2 * R,#    5 > Pair: Pb-00     6px -> Pb-00      6s r = (  3.23,  3.23,  0.00)
        0.38/2 + 0.04/2,#    6 > Pair: Pb-00     6px -> Pb-00     6py r = (  3.23,  3.23,  0.00)
        0.38/2 - 0.04/2,#    7 > Pair: Pb-00     6px -> Pb-00     6px r = (  3.23,  3.23,  0.00)
        0.0, #    8 > Pair: Te-01      5s -> Te-01      5s r = (  0.00,  0.00,  0.00)
        -0.05,#    9 > Pair: Te-01      5s -> Te-01      5s r = (  3.23,  3.23,  0.00)
        -0.4,#   10 > Pair: Te-01      5s -> Pb-00      6s r = (  3.23,  0.00,  0.00)
        -0.77,#   11 > Pair: Te-01      5s -> Pb-00     6px r = (  3.23,  0.00,  0.00)
         8.9, #   12 > Pair: Te-01     5px -> Te-01     5px r = (  0.00,  0.00,  0.00)
        -0.02,  #   13 > Pair: Te-01     5px -> Te-01     5px r = (  0.00,  3.23,  3.23)
        -0.11 * R, #   14 > Pair: Te-01     5px -> Te-01      5s r = (  3.23,  3.23,  0.00)
        0.14/2 + 0.02/2,#   15 > Pair: Te-01     5px -> Te-01     5py r = (  3.23,  3.23,  0.00)
        0.14/2 - 0.02/2,#   16 > Pair: Te-01     5px -> Te-01     5px r = (  3.23,  3.23,  0.00)
        -0.22, #   17 > Pair: Te-01     5px -> Pb-00     6px r = (  0.00,  3.23,  0.00)
        -0.95, #   18 > Pair: Te-01     5px -> Pb-00      6s r = (  3.23,  0.00,  0.00)
         1.50, #   19 > Pair: Te-01     5px -> Pb-00     6px r = (  3.23,  0.00,  0.00)
    ])
"""