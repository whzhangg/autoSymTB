import typing

import numpy as np
from scipy import constants


def calculate_weighted_mobility(
        temp: float, sigma: float, seebeck: float) -> np.ndarray:
    """calculate mobility in SI unit: m^2 / Vs"""
    from scipy import constants
    me = constants.electron_mass
    kb = constants.Boltzmann
    e = constants.elementary_charge
    h = constants.h
    pi = constants.pi 
    S_over_kbe = np.abs(seebeck) / (kb / e)
    bracket = ( np.exp(S_over_kbe - 2) ) / ( 1+np.exp(-5 * (S_over_kbe - 1)) ) \
                + ( 3 * S_over_kbe / pi ** 2 ) / ( 1 + np.exp(5 * (S_over_kbe - 1)) )
    factor = ( 3 * h**3 * sigma) / ( 8 * pi * e * (2 * me * kb * temp)**1.5)
    return factor * bracket


def calculate_seebeck_effectivemass(
        Sij: np.ndarray, ncarrier: float, temp: float) -> float:
    """seebeck effective mass
    
    Implementation of equation (3) in Synder 2022
    """
    kb = constants.Boltzmann
    e = constants.elementary_charge
    h = constants.h
    pi = np.pi 
    n = np.abs(ncarrier)
    s = np.abs(np.trace(Sij) / 3.0 )
    factor1 = h**2 / (2 * kb * temp)
    factor2 = (3 * (n/1.12) / (16 * np.sqrt(pi)))**(2/3)
    term3_up = (np.exp(s/(kb/e)-2) - 0.17)**(2/3)
    term3_dw = 1 + np.exp(-5 * (s/(kb/e) - (kb/e)/s))
    term4_up = (3/pi**2) * (2/np.sqrt(pi))**(2/3) * (s/(kb/e))
    term4_dw = 1 + np.exp( 5 * (s/(kb/e) - (kb/e)/s))
    return (factor1 * factor2 * 
            ( term3_up / term3_dw + term4_up / term4_dw ) / constants.electron_mass)


def calculate_transport(
    mu: float, 
    temp: float, 
    nele: float,
    ek: np.ndarray, 
    vk: np.ndarray, 
    tau: np.ndarray, 
    cell_volume: float, 
    weight: float
) -> typing.Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """calculates transport properties
    
    Notice: kappa = K - T sigma S^2

    Parameters
    ----------
    mu
        chemical potential in eV
    temp
        temperature in K
    nele
        number of electrons, for calculating carrier concentration
    ek: np.ndarray(nk, nbnd)
        eigen energy of electrons in eV
    vk: np.ndarray(nk, nbnd, 3)
        band velocity in m/s
    tau:  np.ndarray(nk, nbnd)
        carrier lifetime is SI
    cell_volume: float
        volume of the cell in A^3
    weight: float
        2 for spin-degenerate calculation

    Returns
    -------
    carrier_concentration
        in unit m^-3
    sigma_ij: np.ndarray(3,3)
        conductivity in S/m
    Sij: np.ndarray(3,3)
        seebeck coefficient in V/k
    k_ij: np.ndarray(3,3)
        electron thermal conductivity W/mK
    """
    nk = len(ek)
    summed_carrier = np.sum(_fd(ek, mu, temp)) * weight / nk 
    #print(summed_carrier)
    carrier_concentration = (summed_carrier - nele) / (cell_volume * 1e-30) # m^{-3}

    vel_tensor = np.einsum("mni, mnj -> mnij", vk, vk)
    dfde_factor = -1 * _dfde(ek, mu, temp)

    sigma_sum = np.zeros((3,3))
    sigmaSsum = np.zeros((3,3))
    K_sum = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            sigma_sum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j])
            sigmaSsum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j] \
                            * (ek - mu) * constants.elementary_charge)
            K_sum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j] * \
                            ((ek - mu) * constants.elementary_charge) **2 )
                
    sigma_sum *= weight
    sigmaSsum *= weight
    K_sum *= weight

    total_volumn = nk * cell_volume * 1e-30 # m3
    sigma_ij = constants.elementary_charge ** 2 * sigma_sum / total_volumn
    sigmaSij = -1 * constants.elementary_charge * sigmaSsum / total_volumn / temp
    k_ij = K_sum / total_volumn / temp

    Sij = np.linalg.inv(sigma_ij) @ sigmaSij
    kappa_ij = k_ij - temp * Sij @ sigma_ij @ Sij

    return carrier_concentration, sigma_ij, Sij, kappa_ij


def _fd(ei,u,T):
    """
    calculate fermi-dirac distribution,
    units of input: ei and u in eV, T in K
    """
    nu = (ei-u) * constants.elementary_charge / (constants.Boltzmann*T)
    nu[np.where(nu > 500)] = 500
    # exp(-) will be zero without overflow
    return 1/( 1+np.exp( nu ) )


def _dfde(ei, u, T):
    """
    SI units
    """
    kbt = constants.Boltzmann * T
    nu = -1 * np.abs(ei-u)* constants.elementary_charge / kbt
    
    return  -1.0 * np.exp(nu) / (1.0 + np.exp(nu))**2 / kbt

