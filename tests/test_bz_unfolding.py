import os, time, numpy as np, tqdm

from automaticTB.solve.functions.solve import solve_interaction
from automaticTB.interface import OrbitalPropertyRelationship
from compare_values import compare_stored_values
from automaticTB.properties import transport

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
    band = model.get_bandstructure(prefix, pbte_path, make_folder=False)
    band.plot_data(f"{prefix}.pdf", yminymax=(8,12))


def test_unfolded_energy(nk = 20):
    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    
    print("calculating_full_grid_energies")
    t0 = time.time()
    k, full_ev = transport.straight_calculation(model.tb, nk)
    print(f" ... time = {time.time() - t0}")

    print("calculate with unfolding")
    t0 = time.time()
    mp = transport.MeshProperties.calculate_from_tb(model.tb, nk)
    unfolded_ev = mp.get_unfolded_energies_velocities()
    print(f" ... time = {time.time() - t0}")
    return
    assert full_ev.shape == unfolded_ev.shape
    nk, nbnd, _ = full_ev.shape
    for ik in range(nk):
        for ibnd in range(nbnd):
            if not np.allclose(full_ev[ik, ibnd, 1:], unfolded_ev[ik, ibnd, 1:], rtol=0.1):
                map_k, map_rot = mp.mapping[ik]
                print(f"[{ik:>4d},{ibnd:>2d}] {map_k:>4d}{map_rot:>2d}", end=" ")
                print("vref = {:>12.4e},{:>12.4e},{:>12.4e}".format(*full_ev[ik, ibnd, 1:]),end = "  ")
                print("v = {:>12.4e},{:>12.4e},{:>12.4e}".format(*unfolded_ev[ik, ibnd, 1:]))


def test_transport(nk):
    from automaticTB.properties import transport
    from automaticTB.properties import kpoints

    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    
    cell = kpoints.UnitCell(model.tb.cell)
    print("calculating_full_grid_energies")
    t0 = time.time()
    k, full_ev = transport.straight_calculation(model.tb, nk)
    print(f" ... time = {time.time() - t0}")
    print(np.max(full_ev[:,:5,0]))
    ncarrier, sigma_ij, Sij, k_ij = transport.calculate_transport(
        mu=9.75, temp=300, nele=10, ek=full_ev[:,:,0], vk=full_ev[:,:,1:], 
        tau=np.ones_like(full_ev[:,:,0]) * 1.0e-13, cell_volume=cell.get_volume(), weight=2
    )

    print(ncarrier / 1e6)
    print(Sij[0,0] * 1e6)
    print(sigma_ij[0,0])
    return


def test_transport_tau(nkgrid):
    from automaticTB.properties import transport
    from automaticTB.properties import kpoints
    from automaticTB.properties import dos

    relationship = OrbitalPropertyRelationship.from_file(result_file)
    retrived_values = get_needed_values(
        [relationship.all_pairs[i] for i in relationship.free_pair_indices])
    model = relationship.get_ElectronicModel_from_free_parameters(
        free_Hijs=retrived_values)
    
    cell = kpoints.UnitCell(model.tb.cell)
    print("calculating_full_grid_energies")
    t0 = time.time()

    uc = kpoints.UnitCell(model.tb.cell)
    mesh = uc.recommend_kgrid(nkgrid)
    dos_mesh = dos.TetraKmesh(uc.reciprocal_cell, mesh)
    mu = 9.75
    e, v = model.tb.solveE_V_at_ks(dos_mesh.kpoints)
    print(f" ... time = {time.time() - t0}")
    dos = dos.TetraDOS(dos_mesh, e, np.linspace(mu-2.0,mu+2.0, 1000))
    dos_x = dos.x
    dos_y = dos.dos

    ph_scattering = np.zeros_like(e)
    nk, nbnd = e.shape
    for ik in tqdm.tqdm(range(nk)):
        for ibnd in range(nbnd):
            if e[ik,ibnd] < mu + 2.0 and e[ik,ibnd] > mu-2.0:
                min_index = np.argmin(np.abs(e[ik,ibnd] - dos_x))
                ph_scattering[ik,ibnd] = max(1e-2, dos_y[min_index])
            else:
                ph_scattering[ik, ibnd] = 1.0

    from matplotlib import pyplot as plt
    f = plt.figure()
    axis = f.subplots()
    axis.set_xlim(mu-2.0,mu+2.0)
    axis.scatter(e.flatten(), ph_scattering.flatten())
    f.savefig("dos.pdf") 

    print(f" ... time = {time.time() - t0}")
    ncarrier, sigma_ij, Sij, k_ij = transport.calculate_transport(
        mu=mu, temp=300, nele=10, ek=e, vk=v, tau=1/ph_scattering, cell_volume=cell.get_volume(), weight=2
    )

    print(ncarrier / 1e6)
    print(Sij[0] * 1e6)
    return

if __name__ == "__main__":
    test_transport(30)
    #test_transport_tau(30)