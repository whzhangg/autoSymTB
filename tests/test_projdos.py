import os, numpy as np

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from automaticTB.properties import reciprocal, bands
from automaticTB.properties import dos
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


def test_solved_values_experimental():
    # action
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = find_interaction_values(
        stored_interaction,
        [relationship.all_pairs[i] for i in relationship.free_pair_indices]
    )
    tb = relationship.get_tightbinding_from_free_parameters(
        free_Hijs=retrived_values, experimental=True)
    kpath = reciprocal.Kpath.from_cell_pathstring(tb.cell, bandpath)
    bs = bands.BandStructure.from_tightbinding_and_kpath(tb, kpath)
    bs.plot_band("test_perovskiteband.pdf")

    rcell = reciprocal.find_RCL(tb.cell)
    nks = reciprocal.recommand_kgrid(rcell, 40)
    tetra_mesh = dos.TetraKmesh(rcell, nks)
    band, coeff = tb.solveE_at_ks(tetra_mesh.kpoints)
    normal_dos = dos.TetraDOS(tetra_mesh, band, 2.0)
    normal_dos_result = normal_dos.calculate_dos(np.linspace(-2.5, 0.2, 50))
    normal_dos_result.plot_dos("test_normal.pdf")

    projdos = dos.TetraProjectDOS(tetra_mesh, band, np.abs(coeff)**2, tb.basis_name, 2.0)
    pdos = projdos.calculate_projdos(np.linspace(-2.5, 0.2, 50))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    axs = fig.subplots()

    plot_name = ['Pb(1) 6s', 'Pb(1) 6py', 'Pb(1) 6pz', 'Pb(1) 6px', 'Cl(2) 3s', 'Cl(2) 3py', 'Cl(2) 3pz', 'Cl(2) 3px',]
    for iname, name in enumerate(plot_name):
        axs.plot(pdos.x, pdos.projdos[:,iname], label = name)
    axs.legend()
    fig.savefig("test_projdos.pdf")

if __name__ == "__main__":
    #solve_interaction(
    #    structure=structure_file,
    #    orbitals_dict=orbitals,
    #    rcut=rcut,
    #    save_filename=result_file,
    #    experimental=True
    #)
    test_solved_values_experimental()