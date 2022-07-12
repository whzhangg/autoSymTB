import numpy as np
import abc, typing, dataclasses

import BoltzTraP2.sphere
import BoltzTraP2.fite
import BoltzTraP2.bandlib
from BoltzTraP2.units import Angstrom, Meter, eV

eV2Hartree = eV # eV is from BoltzTrap
from .kmesh import Kmesh
from ..tightbinding import TightBindingBase
from ..tools import find_RCL, atom_from_cpt,  write_yaml

# TODO: write test for BoltzTrap Data

@dataclasses.dataclass
class SingleBoltzTrapResult:
    """
    single condition transport property, 
    Important thing to note here is that sigma is provided as per second quantity if using 
    single relaxation time approximation 
    """
    temperature: float
    mu: float
    seebeck: np.ndarray # 3*3
    sigma: np.ndarray # 3*3
    ncarrier: float
    units: typing.Dict[str, str]


class BoltzTrapData(abc.ABC):
    WINDOW = 0.26 # Hartree ~ 7 eV

    @property
    @abc.abstractmethod
    def kpoints(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def ebands(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def mommat(self) -> typing.Optional[np.ndarray]:
        return None

    
    @abc.abstractmethod
    def get_lattvec(self) -> np.ndarray:
        raise NotImplementedError

    
    def select_band(self, ebands: np.ndarray, efermi_eV: float) -> typing.List[int]:
        efermi = efermi_eV / eV2Hartree
        nbnd = len(ebands)
        up = efermi + self.WINDOW
        down = efermi - self.WINDOW
        indices = []
        for ibnd in range(nbnd):
            if (ebands[ibnd] < down).all() or (ebands[ibnd] > up).all():
                continue

            indices.append(ibnd)

        return indices


class TBoltzTrapData():
    WINDOW = 0.26 # Hartree ~ 7 eV
    def __init__(self, kpoints: np.ndarray, ebnds: np.ndarray, cell: np.ndarray, approximate_fermi_eV: float) -> None:
        assert len(kpoints) == ebnds.shape[1]
        self._kpoints = kpoints
        include_band = self.select_band(ebnds, approximate_fermi_eV)
        self._ebnds = ebnds[include_band]
        self._cell = cell

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def ebands(self) -> np.ndarray:
        return self._ebnds * eV2Hartree

    @property
    def mommat(self) -> typing.Optional[np.ndarray]:
        return None

    def get_lattvec(self) -> np.ndarray:
        # see line 63, BoltzTrap2/examples/parabolic.py
        return self._cell.T * Angstrom

    def select_band(self, ebands: np.ndarray, efermi_eV: float) -> typing.List[int]:
        efermi = efermi_eV / eV2Hartree
        nbnd = len(ebands)
        up = efermi + self.WINDOW
        down = efermi - self.WINDOW
        indices = []
        for ibnd in range(nbnd):
            if (ebands[ibnd] < down).all() or (ebands[ibnd] > up).all():
                continue

            indices.append(ibnd)

        return indices

# the code feels a bit ad-hoc, since I did not dig into the internals of BoltzTrap

class TBoltzTrapCalculation:
    def __init__(self, 
        tightbinding: TightBindingBase,
        ngrid: typing.Tuple[int, int, int],
        nele: int,
        dosweight: int
    ) -> None:
        self._tightbinding = tightbinding
        self._nele = nele
        self._dosweight = dosweight

        self._kmesh = Kmesh(
            find_RCL(self._tightbinding.cell), ngrid
        )
        self._energies = self._tightbinding.solveE_at_k(self._kmesh.kpoints).T  # so that we have [nbnd, nk]

        _sorted_eigenvalue = np.sort(self._energies.flatten())
        which = self._nele * self._kmesh.numk // self.dosweight
        self._approximate_ef = _sorted_eigenvalue[which] # in eV

        self._boltztrapData = TBoltzTrapData(
            self._kmesh.kpoints, self._energies, self._tightbinding.cell, self._approximate_ef
        )

        self._lastresults: typing.List[SingleBoltzTrapResult] = []


    def calculate(
        self, temps: typing.List[float], mu: typing.List[float], factor: int
    ) -> typing.List[SingleBoltzTrapResult]:
        temps = np.array(temps)
        murange = np.array(mu) * eV2Hartree

        structure = atom_from_cpt(
            self._tightbinding.cell, self._tightbinding.positions, self._tightbinding.types
        )


        equivalences = BoltzTraP2.sphere.get_equivalences(structure, None,
                                                        factor * self._kmesh.numk)

        coeffs = BoltzTraP2.fite.fitde3D(self._boltztrapData, equivalences)

        eband, vvband, cband = BoltzTraP2.fite.getBTPbands(equivalences, coeffs,
                                                        self._boltztrapData.get_lattvec())
        
        dose, dos, vvdos, cdos = BoltzTraP2.bandlib.BTPDOS(eband, vvband, 
                                    erange=(self._approximate_ef - 1.0, self._approximate_ef + 1.0), npts=2000)

        
        N, L0, L1, L2, Lm11 = BoltzTraP2.bandlib.fermiintegrals(
                            dose, dos, vvdos, mur=murange, Tr = temps, dosweight= self._dosweight)

        volume = np.linalg.det(self._boltztrapData.get_lattvec())
        sigma, seebeck, kappa, Hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(L0, L1, L2, murange, temps, volume)
        # sigma is calculated with tau = 1.0
        
        # N gives the tot number of electrons
        dN = (-N) * self._dosweight  - self._nele   # positive for more electrons, n type
        dN = dN / (volume / (Meter / 100.)**3)  

        results: typing.List[SingleBoltzTrapResult] = []
        for itemp, temp in enumerate(temps):
            for imu, mu in enumerate(murange):
                results.append(
                    SingleBoltzTrapResult(
                        temperature = temp,
                        mu = mu,
                        seebeck = seebeck[itemp, imu],
                        sigma = sigma[itemp, imu],
                        ncarrier = dN[itemp, imu],
                        units = {
                            "temperature": "K",
                            "mu": "eV",
                            "seebeck": "V/K",
                            "sigma": "S/(ms)",
                            "ncarrier": "cm^{-3}"
                        }
                    )
                )
        
        self._lastresults = results
        return results


    def write_last_results_to_yaml(self, filename: str) -> None:
        dicted_result = [ dataclasses.asdict(r) for r in self._lastresults ]
        write_yaml(dicted_result, filename)


@dataclasses.dataclass
class TBoltzTrapCalculation_previous:
    tightbinding: TightBindingBase
    ngrid: typing.Tuple[int, int, int]
    nele: int
    dosweight: int

    def _get_approximate_fermi(self, ebnds: np.ndarray) -> float:
        """
        using 
        """
        Nk = self.ngrid[0] * self.ngrid[1] * self.ngrid[2]
        sorted_eigenvalue = np.sort(ebnds.flatten())
        which = self.nele * Nk // self.dosweight
        return sorted_eigenvalue[which]

    @property
    def kpoints(self) -> np.ndarray:
        reciprocal_cell = find_RCL(self.tightbinding.cell)
        kmesh = Kmesh(reciprocal_cell, self.ngrid)
        return kmesh.kpoints

    def _get_boltztrap_data(self) -> TBoltzTrapData:
        kpoints = self.kpoints
        energies = self.tightbinding.solveE_at_ks(kpoints).T  # nbnd, nk
        approximate_ef = self.get_approximate_fermi(energies)
        return TBoltzTrapData(
            kpoints, energies, self.tightbinding.cell, approximate_ef
        )
    
    def calculate(self, 
        temps: typing.List[float], 
        mu: typing.List[float],
        factor: int 
    ) -> dict:

        temps = np.array(temps)
        murange = np.array(mu) * eV2Hartree
        degeneracy = self.dosweight

        structure = atom_from_cpt(
            self.tightbinding.cell, self.tightbinding.positions, self.tightbinding.types
        )

        data = self._get_boltztrap_data()
        
        # although data store dosweight, just like DFTdata class in boltztrap2/dft.py
        # it is seems not directly used by the Boltztrap2
        # rather, it is passed to dosweight (default to 2.0)
        # see examples/Bi2Te3_Kappa.py 

        mu0 = self.get_approximate_fermi(data.ebands)

        nkp_nscf = len(data.kpoints)

        equivalences = BoltzTraP2.sphere.get_equivalences(structure, None,
                                                        factor * nkp_nscf)

        coeffs = BoltzTraP2.fite.fitde3D(data, equivalences)

        eband, vvband, cband = BoltzTraP2.fite.getBTPbands(equivalences, coeffs,
                                                        data.get_lattvec())
        
        dose, dos, vvdos, cdos = BoltzTraP2.bandlib.BTPDOS(eband, vvband, 
                                    erange=(mu0 - 1.0, mu0 + 1.0), npts=2000)

        
        N, L0, L1, L2, Lm11 = BoltzTraP2.bandlib.fermiintegrals(
                            dose, dos, vvdos, mur=murange, Tr = temps, dosweight= degeneracy)

        volume = np.linalg.det(data.get_lattvec())
        sigma, seebeck, kappa, Hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(L0, L1, L2, murange, temps, volume)
        # sigma is calculated with tau = 1.0
        
        # N gives the tot number of electrons
        dN = (-N) * degeneracy  - self.nele   # positive for more electrons, n type
        dN = dN / (volume / (Meter / 100.)**3)  

        results: typing.List[SingleBoltzTrapResult] = []
        for itemp, temp in enumerate(temps):
            for imu, mu in enumerate(murange):
                results.append(
                    SingleBoltzTrapResult(
                        temperature = temp,
                        mu = mu,
                        seebeck = seebeck[itemp, imu],
                        sigma = sigma[itemp, imu],
                        ncarrier = dN[itemp, imu],
                        units = {
                            "temperature": "K",
                            "mu": "eV",
                            "seebeck": "V/K",
                            "sigma": "S/(ms)",
                            "ncarrier": "cm^{-3}"
                        }
                    )
                )
        

        return results


def write_results_yaml(results: typing.List[SingleBoltzTrapResult]) -> None:
    dicted_result = [ dataclasses.asdict(r) for r in results ]
    write_yaml(dicted_result, "boltztrap2.yaml")