import numpy as np
import abc, typing, dataclasses

import BoltzTraP2.sphere
import BoltzTraP2.fite
import BoltzTraP2.bandlib
from BoltzTraP2.units import Angstrom, Meter, eV

eV2Hartree = eV # eV is from BoltzTrap
from .dos import Kmesh
from ..tightbinding import TightBindingBase
from ..tools import find_RCL, atom_from_cpt, write_json

# TODO: write test for BoltzTrap Data

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


class TBoltzTrapData(BoltzTrapData):
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

    def get_lattvec(self) -> np.ndarray:
        # see line 63, BoltzTrap2/examples/parabolic.py
        return self._cell.T * Angstrom


@dataclasses.dataclass
class TBoltzTrapCalculation:
    tightbinding: TightBindingBase
    ngrid: typing.Tuple[int, int, int]
    nele: int
    dosweight: int

    def get_approximate_fermi(self, ebnds: np.ndarray) -> float:
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
        temp: typing.List[float], 
        mu: typing.List[float],
        tau_sec: float, 
        factor: int
    ) -> dict:

        temp = np.array(temp)
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
                            dose, dos, vvdos, mur=murange, Tr = temp, dosweight= degeneracy)

        volume = np.linalg.det(data.get_lattvec())
        sigma, seebeck, kappa, Hall = BoltzTraP2.bandlib.calc_Onsager_coefficients(L0, L1, L2, murange, temp, volume)

        sigma *= tau_sec
        # N gives the tot number of electrons
        dN = (-N) * degeneracy  - self.nele   # positive for more electrons, n type
        dN = dN / (volume / (Meter / 100.)**3)  

        final = {}
        
        final["units"] = {"temperature" : "K",
                        "chemical_potential" : "eV",
                        "relaxation_time" : "second",
                        "seebeck" : "V/K",
                        "sigma" : "S/m",
                        "carrier_concentration" : "cm^{-3}"
                        }

        final["temperature"] = temp
        final["chemical_potential"] = murange / eV2Hartree
        final["relaxation_time"] = tau_sec
        final["seebeck"] = seebeck
        final["sigma"] = sigma
        final["carrier_concentration"] = dN

        return final


def write_boltztrap_result2file(filename, result):
    result["temperature"] = result["temperature"].tolist()
    result["chemical_potential"] = result["chemical_potential"].tolist()
    result["seebeck"] = result["seebeck"].tolist()
    result["sigma"] = result["sigma"].tolist()
    result["carrier_concentration"] = result["carrier_concentration"].tolist()
    write_json(result, filename)
