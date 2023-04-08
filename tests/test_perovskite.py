import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from automaticTB.properties import reciprocal
from utilities import find_interaction_values

prefix = "perovskite"
orbitals = {"Pb":"6s6p", "Cl": "3s3p"}
rcut = 3.0
structure_file = os.path.join("..", "resources", "perovskite.cif")
bandpath = [
    "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
    "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
    "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
    "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
    "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
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
    tb = relationship.get_tightbinding_from_free_parameters(
        free_Hijs=retrived_values)
    #unitcell = kpt.UnitCell(model.tb.cell)
    #kpath = unitcell.get_kpath_from_path_string(bandpath)
    kpath = reciprocal.Kpath.from_cell_pathstring(tb.cell, bandpath)
    values, _ = tb.solveE_at_ks(kpath.kpoints)
    stacked = np.hstack([kpath.kpoints, values])
    # test
    compare = np.load(stored_result)
    assert np.allclose(stacked, compare, atol = 1e-3)
    os.remove(result_file)


if __name__ == "__main__":
    from automaticTB.properties import bands
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
    tb = relationship.get_tightbinding_from_free_parameters(
        free_Hijs=retrived_values)

    kpath = reciprocal.Kpath.from_cell_pathstring(tb.cell, bandpath)
    bs = bands.BandStructure.from_tightbinding_and_kpath(tb, kpath, order_band=False)
    bs.plot_band(f"{prefix}.pdf")