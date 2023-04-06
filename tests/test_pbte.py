import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from automaticTB.properties import kpoints as kpt
from utilities import find_interaction_values

prefix = "pbte"
orbitals = {"Pb":"6s6p", "Te": "5s5p"}
rcut = 4.7
structure_file = os.path.join("..", "resources", "PbTe_ICSD38295.cif")
bandpath = pbte_path = [
    'G 0.0000 0.0000 0.0000 X 0.5000 0.0000 0.5000', 
    'X 0.5000 0.0000 0.5000 W 0.5000 0.2500 0.7500', 
    'W 0.5000 0.2500 0.7500 L 0.5000 0.5000 0.5000', 
    'L 0.5000 0.5000 0.5000 G 0.0000 0.0000 0.0000', 
    'G 0.0000 0.0000 0.0000 K 0.3750 0.3750 0.7500'
]

result_file = f"{prefix}.result"
stored_interaction = f"reference/{prefix}.stored_interactions"
stored_result = f"reference/{prefix}.npy"


def test_solved_values():
    # setup
    solve_interaction(
        structure=structure_file,
        orbitals_dict=orbitals,
        rcut=rcut,
        save_filename=result_file
    )
    # action
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = find_interaction_values(
        stored_interaction,
        [relationship.all_pairs[i] for i in relationship.free_pair_indices]
    )
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    unitcell = kpt.UnitCell(model.tb.cell)
    kpath = unitcell.get_kpath_from_path_string(bandpath)
    values = model.tb.solveE_at_ks(kpath.kpoints)
    stacked = np.hstack([kpath.kpoints, values])
    # test
    compare = np.load(stored_result)
    assert np.allclose(stacked, compare, atol = 1e-3)
    os.remove(result_file)


if __name__ == "__main__":
    solve_interaction(
        structure=structure_file,
        orbitals_dict=orbitals,
        rcut=rcut,
        save_filename=result_file
    )
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = find_interaction_values(
        stored_interaction,
        [relationship.all_pairs[i] for i in relationship.free_pair_indices]
    )
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)

    bandresult = model.get_bandstructure(prefix, bandpath, make_folder=False)
    bandresult.plot_data(f"{prefix}.pdf")