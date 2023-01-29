import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship

prefix = "pbte"
result_file = f"{prefix}.result"

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
    kpoints = {
        "G": np.zeros(3, dtype=float),
        "X": np.array([1/2, 0.0, 1/2]),
        "W": np.array([1/2, 1/4, 3/4]),
        "L": np.array([1/2, 1/2, 1/2]),
        "K": np.array([3/8, 3/8, 3/4])
    }

    path_str = [
        "G {:>.4f} {:>.4f} {:>.4f} X {:>.4f} {:>.4f} {:>.4f}".format(*kpoints["G"], *kpoints["X"]),
        "X {:>.4f} {:>.4f} {:>.4f} W {:>.4f} {:>.4f} {:>.4f}".format(*kpoints["X"], *kpoints["W"]),
        "W {:>.4f} {:>.4f} {:>.4f} L {:>.4f} {:>.4f} {:>.4f}".format(*kpoints["W"], *kpoints["L"]),
        "L {:>.4f} {:>.4f} {:>.4f} G {:>.4f} {:>.4f} {:>.4f}".format(*kpoints["L"], *kpoints["G"]),
        "G {:>.4f} {:>.4f} {:>.4f} K {:>.4f} {:>.4f} {:>.4f}".format(*kpoints["G"], *kpoints["K"]),
        ]
    
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
    model = relationship.get_ElectronicModel_from_free_parameters(free_Hijs=values)
    print(f"# Band structure will be plotted with prefix {prefix}")
    print("")
    model.plot_bandstructure(
        prefix = prefix,
        kpaths_str = path_str,
        yminymax=(8,12)
    )

    #print(f"Calculate effective mass at CBM")
    #mass = BandEffectiveMass(model.tb)
    #cb = mass.calc_effective_mass(np.array([0.367, 0.0, 0.367]), ibnd=4)
    #print(cb)

extract()