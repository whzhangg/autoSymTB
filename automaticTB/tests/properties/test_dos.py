import numpy as np
import typing
from automaticTB.properties.dos import TetraDOS, Kmesh
from automaticTB.tightbinding.RoyerRichardTB import Royer_Richard_TB


def prepare_RR_tightbinding_kmesh() -> typing.Tuple[Royer_Richard_TB, Kmesh]:
    reciprocal_cell = np.eye(3)
    ngrid = [10,10,10]

    kmesh = Kmesh(reciprocal_cell, ngrid)
    tb = Royer_Richard_TB()

    return tb, kmesh


def prepare_singleband_energy_kmesh() -> typing.Tuple[np.ndarray, Kmesh]:
    cell = np.array([5.0,5.0,5.0]) 
    reciprocal_cell = np.diag( 2 * np.pi / cell )  # 1/a0
    ngrid = [20] * 3
    kmesh = Kmesh(reciprocal_cell, ngrid)
    energies = []
    em = 1.0
    for k in kmesh.kpoints:
        xk = reciprocal_cell.dot(np.array([0.5,0.5,0.5]) - k)
        energies.append( np.linalg.norm(xk)**2 / 2.0 / em )
    energies = np.vstack(energies)
    assert energies.shape == (kmesh.numk, 1)
    
    return energies, kmesh


def test_Royer_Richard_dos():
    tb, kmesh = prepare_RR_tightbinding_kmesh()
    energies = tb.solveE_at_ks(kmesh.kpoints)

    tetrados = TetraDOS(kmesh, energies, np.linspace(-17.0, 10.0, 50))
    tetrados.write_data_to_file("DOS_RoyerRichardPerovskite.dat")
    tetrados.plot_data("DOS_RoyerRichardPerovskite.pdf")


def test_single_band_dos():
    energy, kmesh = prepare_singleband_energy_kmesh()
    tetrados = TetraDOS(kmesh, energy, np.arange(-0.03, 0.6, 0.01))
    tetrados.write_data_to_file("DOS_parabolic.dat")
    tetrados.plot_data("DOS_parabolic.pdf")
