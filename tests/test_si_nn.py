import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from utilities import find_interaction_values
from automaticTB.properties import reciprocal

prefix = "si_nn"
orbitals = {"Si":"3s3p4s"}
rcut = None
structure_file = os.path.join("..", "resources", "Si_ICSD51688.cif")
bandpath = [
    "L   0.5   0.5  0.5  G  0.0   0.0  0.0",
    "G   0.0   0.0  0.0  X  0.5   0.0  0.5",
    "X   0.5   0.0  0.5  U  0.625 0.25 0.625",
    "K 0.375 0.375 0.75  G  0.0   0.0  0.0"
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
    #model = relationship.get_ElectronicModel_from_free_parameters(
    #    free_Hijs=retrived_values)
    tb = relationship.get_tightbinding_from_free_parameters(
        free_Hijs=retrived_values
    )
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
    from automaticTB.properties import BandStructure
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
    bs = BandStructure.from_tightbinding_and_kpath(tb, kpath, order_band=True)
    bs.plot_band(f"{prefix}.pdf")
    bs.plot_fatband(f"{prefix}_fatband.pdf", {"Si s": ["Si(1) 4s", "Si(2) 4s"]})

    kmesh = reciprocal.Kmesh.from_cell_nk(tb.cell, 20)
    from automaticTB.properties import dos
    from automaticTB.properties import calc_mesh
    tkmesh = dos.TetraKmesh(kmesh.reciprocal_lattice, kmesh.nks)
    e, v = calc_mesh.calculate_e_v_using_ibz(tb, tkmesh.kpoints, tkmesh.nks)
    doscal = dos.TetraDOS(tkmesh, e)
    dos_r = doscal.calculate_dos(np.linspace(-5, 5, 100))
    dos_r.plot_dos(f"{prefix}_dos.pdf")