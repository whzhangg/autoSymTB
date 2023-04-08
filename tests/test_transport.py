import numpy as np
import dataclasses
import pytest
from scipy.optimize import curve_fit

from automaticTB.properties import tightbinding
from automaticTB.properties import bands, dos, transport, calc_mesh
from automaticTB.properties import reciprocal

prefix = "singleband"
temp = 300
mesh_nk = 50
bandpath = [
    "M 0.5 0.5 0.0  R 0.5 0.5 0.5",
    "R 0.5 0.5 0.5  G 0.0 0.0 0.0",
    "G 0.0 0.0 0.0  X 0.5 0.0 0.0",
    "X 0.5 0.0 0.0  M 0.5 0.5 0.0",
    "M 0.5 0.5 0.0  G 0.0 0.0 0.0"
]


def fit_dos(x, a):
    return a * np.sqrt(x)


def seebeck_from_eta(mu, t, l):
    """seebeck in SI (V/K)"""
    from scipy.integrate import quad
    from scipy.constants import Boltzmann, elementary_charge

    def fermi_integrand(e, eta, j):
        return e**j / (1 + np.exp(e - eta))

    def F(eta, j):
        e_upper = eta + 100
        return quad(fermi_integrand, 0, e_upper, args=(eta, j))[0]

    eta = mu * elementary_charge / t / Boltzmann
    F_up = F(eta, 1+l)
    F_dw = F(eta, l)
    return (Boltzmann / elementary_charge) * ((2+l) * F_up / (1 + l) / F_dw - eta)


@dataclasses.dataclass
class TransportSetup:
    sbtb: tightbinding.TightBindingModel
    doscalc: dos.TetraDOS
    e: np.ndarray
    v: np.ndarray


@pytest.fixture(scope="module")
def transport_setup() -> TransportSetup:
    sbtb = tightbinding.get_singleband_tightbinding_with_overlap(overlap=0)
    #kpath = reciprocal.Kpath.from_cell_pathstring(sbtb.cell, bandpath)
    #b = bands.BandStructure.from_tightbinding_and_kpath(sbtb, kpath)
    #b.plot_band(f"{prefix}_band.pdf")
    nks = reciprocal.recommand_kgrid(reciprocal.find_RCL(sbtb.cell), mesh_nk)
    tetramesh = dos.TetraKmesh(reciprocal.find_RCL(sbtb.cell), nks)
    e, v = calc_mesh.calculate_e_v_using_ibz(sbtb, tetramesh.kpoints, tetramesh.nks)
    doscalc = dos.TetraDOS(tetramesh, e, 2.0)

    return TransportSetup(
        sbtb, doscalc, e, v
    )


def test_single_band_lorentz(transport_setup: TransportSetup):
    sbtb = transport_setup.sbtb
    e = transport_setup.e
    v = transport_setup.v

    emax = np.max(e)
    nc, sigma_ij, Sij, k_ij = transport.calculate_transport(
        emax - 2.0, temp, 2.0, e, v, np.ones_like(e), reciprocal.get_volume(sbtb.cell), 2.0
    )
    lorentz = k_ij[0,0] / sigma_ij[0,0] / temp
    print(f"Lorentz number: {lorentz}")
    assert np.isclose(lorentz, 2.44e-8, rtol = 0.02)


def test_single_band_seebeck_crt(transport_setup: TransportSetup):
    sbtb = transport_setup.sbtb
    e = transport_setup.e
    v = transport_setup.v

    # kpath = reciprocal.Kpath.from_cell_pathstring(sbtb.cell, bandpath)
    # b = bands.BandStructure.from_tightbinding_and_kpath(sbtb, kpath)
    # b.plot_band(f"{prefix}_band.pdf")
    emax = np.max(e)

    #doscalc = transport_setup.doscalc
    #dos_range = np.linspace(np.min(e)-0.1, np.max(e)+0.1, 200)
    #dosresult = doscalc.calculate_dos(dos_range)
    #dosresult.plot_dos(f"{prefix}_dos.pdf")
    #print(f"Num ele: {doscalc.nsum(20)}")

    # check seebeck values
    ## constant tau
    mus = np.linspace(emax - 0.2, emax, 10)
    ref_seebeck_crt = np.array([seebeck_from_eta(emax-mu, temp, 0.5) for mu in mus])
    cal_seebeck_crt = np.zeros_like(ref_seebeck_crt)
    for imu, mu in enumerate(mus):
        _,_, Sij, _ = transport.calculate_transport(
            mu, temp, 2.0, e, v, np.ones_like(e), reciprocal.get_volume(sbtb.cell), 2.0
        )
        cal_seebeck_crt[imu] = Sij[0,0]
        
    print("Constant tau")
    for mu, ref, cal in zip(mus, ref_seebeck_crt, cal_seebeck_crt):
        print(f"{mu:>8.3f} ref={ref*1e6:>8.3f} cal={cal*1e6:>8.3f}")
    assert np.allclose(ref_seebeck_crt, cal_seebeck_crt, atol=5.0)


def test_single_band_seebeck_invg(transport_setup: TransportSetup):
    sbtb = transport_setup.sbtb
    e = transport_setup.e
    v = transport_setup.v
    doscalc = transport_setup.doscalc

    emax = np.max(e)
    dos_range = np.linspace(np.min(e)-0.1, np.max(e)+0.1, 200)
    dosresult = doscalc.calculate_dos(dos_range)
    
    fit_window = 1.0
    fit_x = emax - dosresult.x
    musk = np.logical_and(fit_x > 0.0, fit_x < fit_window)
    fit_x = fit_x[musk]
    fit_y = dosresult.dos[musk]

    fit, _ = curve_fit(fit_dos, fit_x, fit_y)
    a = fit[0]

    mus = np.linspace(emax - 0.2, emax, 10)
    
    ref_seebeck_mfp = np.array([seebeck_from_eta(emax-mu, temp, 0.0) for mu in mus])
    cal_seebeck_mfp = np.zeros_like(ref_seebeck_mfp)

    tau = np.zeros_like(e)
    nk, nbnd = e.shape
    for ik in range(nk):
        for ibnd in range(nbnd):
            tau[ik, ibnd] = 1.0 / (np.max([1e-10, fit_dos(emax - e[ik, ibnd], a)]))

    for imu, mu in enumerate(mus):
        _, _, Sij, _ = transport.calculate_transport(
            mu, temp, 2.0, e, v, tau, reciprocal.get_volume(sbtb.cell), 2.0
        )
        cal_seebeck_mfp[imu] = Sij[0,0]
    print("")
    print("tau ~ inv(g)")
    for mu, ref, cal in zip(mus, ref_seebeck_mfp, cal_seebeck_mfp):
        print(f"{mu:>8.3f} ref={ref*1e6:>8.3f} cal={cal*1e6:>8.3f}")
    assert np.allclose(ref_seebeck_mfp, cal_seebeck_mfp, atol=5.0)


