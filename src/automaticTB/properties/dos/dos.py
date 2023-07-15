import dataclasses
import typing

import numpy as np

from automaticTB import tools
from .tetra_mesh import TetraKmesh


def find_vbm_with_gap(doscalc: "TetraDOS", nele: float, resolution: float = 0.01) -> float:
    """find the chemical potential from DOS at 0 K
    
    experimental, not reliable
    """
    emin = np.min(doscalc._mesh_energies) - 0.1
    emax = np.max(doscalc._mesh_energies) + 0.1

    dosresult_coarse = doscalc.calculate_dos(np.linspace(emin, emax, 200))
    
    for isum, csum in enumerate(dosresult_coarse.accumlatedos):
        if csum > nele - 0.1: 
            emin = dosresult_coarse.x[isum-1]
            break
    
    e = emin
    while True:
        e += resolution
        if e > emax: break
        sumdos = doscalc.nsum(e)
        print(e, sumdos)
        if np.isclose(sumdos, nele, atol = 1e-4):
            break

    return e


@dataclasses.dataclass
class DosResult:
    x: np.ndarray
    dos: np.ndarray

    @property
    def accumlatedos(self) -> np.ndarray:
        #https://numpy.org/doc/stable/reference/generated/numpy.ufunc.accumulate.html
        x2 = np.roll(self.x, -1)
        x2[-1] = x2[-2]*2 - x2[-3]
        dx = x2 - self.x
        return np.add.accumulate(self.dos * dx)


    def write_data_to_file(self, filename: str):
        assert self.x.shape == self.dos.shape
        with open(filename, 'w') as f:
            f.write("#{:>19s}{:>20s}".format("Energy (eV)", "DOS (1/eV)"))
            for x, y in zip(self.x, self.dos):
                f.write("{:>20.12e}{:>20.12e}\n".format(x,y))


    def plot_dos(self, filename: str):
        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_xlabel("Energy (eV)")
        axes.set_ylabel("Density of States (1/eV)")

        ymax = np.max(self.dos) * 1.2
        ymin = 0
        axes.set_xlim(min(self.x), max(self.x))
        axes.set_ylim(ymin, ymax)

        axes.plot(self.x, self.dos)

        fig.savefig(filename)


@dataclasses.dataclass
class ProjDosResult:
    x: np.ndarray
    orbname: typing.List[str]
    projdos: np.ndarray


class TetraDOS:
    """Tetrahedron DOS

    It provide x (eV), dos (1/eV) for plotting the density of state.
    It can also return DOS at given energy (`single_dos`) or the 
    sum of number of electrons below the given energy (`nsum)

    It require a `TetraKmesh` object and energies calculated for the 
    kpoints on the mesh. So it's better not to create manually. 

    This tetrahedron method is adopted from:
    https://github.com/tflovorn/tetra
    Theory:
        k point density in the reciprocal space is given by: 
        V / (8 * pi)^3
        where V is the volume of the crystal: V = N * V_cell

        The density of state of the whole system is given by:

        G(E) = weight * [V / (4 * pi^2 ) ( 2m* / hbar )^{3/2} sqrt(E)]
        
        where weight = 1, 2 the spin degeneracy

        The density of state per unit cell, which is calculated by the 
        dos() method provided here, is:

        g(E) = G(E) / N = 
        weight * [ V_cell / (4 * pi^2 ) ( 2m* / hbar )^{3/2} sqrt(E) ]

        nsum is the summation of number of states in the BZ, per unit 
        cell below a given energy.
        nsum is related to the dos calculated here (dos()) by:
            n = int_{-\infty}^{E} g(E') dE'
        it is given by:
            n = weight * [ 1/N \sum_{k; Ek < E} ] 
        which should equal to the number of electrons per unit cell.
    """
    parallel_threshold = 8000

    def __init__(self, 
        kmesh: TetraKmesh, 
        mesh_energies: np.ndarray,
        dos_weight: float
    ):
        """setup the dos calculation

        Parameters
        ----------
        kmesh: TetraKmesh
            the tetragonal kmesh for summation
        mesh_energies
            eigen-energies calculated for kmesh.kpoints
        
        """
        nk, self._nbnd = mesh_energies.shape
        if nk != kmesh.numk:
            raise "run energies with kmesh!"

        self._kmesh = kmesh
        self._mesh_energies = mesh_energies.T.copy()
        self._dos_weight = dos_weight
        

    def calculate_dos(self, dos_energies: np.ndarray) -> DosResult:
        """calculate dos for the input array of energies
        
        result in unit 1/eV
        """
        result = np.zeros_like(dos_energies)
        #for ie, e in enumerate(self._dos_energies):
        #    result[ie] = self.single_dos(e)
        for ibnd in range(self._nbnd):
            if (np.min(self._mesh_energies[ibnd]) > np.max(dos_energies) or 
                np.max(self._mesh_energies[ibnd]) < np.min(dos_energies)):
                continue
            result += tools.cython_dos_contribution_multiple(
                self._mesh_energies[ibnd], self._kmesh.tetras, dos_energies)
        
        return DosResult(dos_energies, result*self._dos_weight)


    def nsum(self, e: float) -> dict:
        # sum the number of states with energy below E
        nsum = 0.0

        for ibnd in range(self._nbnd):
            nsum += tools.cython_nsum_contribution(
                self._mesh_energies[ibnd], self._kmesh.tetras, e)
            
        nsum *= self._dos_weight
        return nsum


    def single_dos(self, e: float) -> np.ndarray:
        # summed contribution from all bands at energy e
        dos = 0.0

        for ibnd in range(self._nbnd):
            dos += tools.cython_dos_contribution(
                self._mesh_energies[ibnd], self._kmesh.tetras, e)

        dos *= self._dos_weight
        return dos


    def _dos_contribution(self, e: float, band_index: int) -> float:
        """no longer used since we use cython version"""
        # return the density of state at energy E from ith band 
        # taken from tetra/dos.py/DosContrib()

        assert band_index < self._mesh_energies.shape[1]

        sum_dos = 0.0
        num_tetra = len(self._kmesh.tetras)

        for tetra in self._kmesh.tetras:
            e1, e2, e3, e4 = sorted(self._mesh_energies[tetra, band_index])

            if e <= e1:
                sum_dos += 0.0

            elif e1 <= e <= e2:
                if e1 == e2:
                    sum_dos += 0.0
                else:
                    sum_dos += (1/num_tetra) * 3*(e - e1)**2/((e2 - e1)*(e3 - e1)*(e4 - e1))

            elif e2 <= e <= e3:
                if e2 == e3:
                    sum_dos += (1.0 / num_tetra) * 3.0 * (e2 - e1) / ((e3 - e1) * (e4 - e1))
                else:
                    fac = (1/num_tetra) / ((e3 - e1)*(e4 - e1))
                    elin = 3*(e2 - e1) + 6*(e - e2)
                    esq = -3*(((e3 - e1) + (e4 - e2))/((e3 - e2)*(e4 - e2))) * (e - e2)**2
                    sum_dos += fac * (elin + esq)

            elif e3 <= e <= e4:
                if e3 == e4:
                    sum_dos += 0.0
                else:
                    sum_dos += (1/num_tetra) * 3*(e4 - e)**2/((e4 - e1)*(e4 - e2)*(e4 - e3))

            else:
                # E >= E4
                sum_dos += 0.0

        return sum_dos


    def _sum_contribution(self, e: float, band_index: int) -> float:
        """no longer used since we use cython version"""
        assert band_index < self._mesh_energies.shape[1]

        nsum = 0.0
        num_tetra = len(self._kmesh.tetras)

        for tetra in self._kmesh.tetras:
            e1, e2, e3, e4 = sorted(self._mesh_energies[tetra, band_index])
            if e <= e1:
                nsum += 0.0

            elif e1 <= e <= e2:
                if e1 == e2:
                    nsum += 0.0
                else:
                    nsum += (1.0/num_tetra) * (e - e1)**3 / ((e2 - e1)*(e3 - e1)*(e4 - e1))

            elif e2 <= e <= e3:
                if e2 == e3:
                    nsum += (1.0 / num_tetra) * (e2 - e1) * (e2 - e1) / ((e3 - e1) * (e4 - e1))
                else:
                    fac = (1.0/num_tetra) / ((e3 - e1)*(e4 - e1))
                    esq = (e2 - e1)**2 + 3.0*(e2 - e1)*(e - e2) + 3.0*(e - e2)**2
                    ecub = -(((e3 - e1) + (e4 - e2))/((e3 - e2)*(e4 - e2))) * (e - e2)**3
                    nsum += fac * (esq + ecub)

            elif e3 <= e <= e4:
                if e3 == e4:
                    nsum += 1.0 / num_tetra
                else:
                    nsum += (1.0/num_tetra) * (1.0 - (e4 - e)**3/((e4 - e1)*(e4 - e2)*(e4 - e3)))

            else:
                # E >= E4
                nsum += (1.0/num_tetra)

        return nsum


class TetraProjectDOS:
    """Tetrahedron DOS

    it is almost the same to normal TetraDOS but for each tetrahedron, 
    the calculated DOS is multiplied by the average DOS weight
    """
    parallel_threshold = 8000

    def __init__(self, 
        kmesh: TetraKmesh, 
        mesh_energies: np.ndarray,
        mesh_components: np.ndarray,
        orbital_names: typing.List[str],
        dos_weight: float
    ):
        """setup the dos calculation

        Parameters
        ----------
        kmesh: TetraKmesh
            the tetragonal kmesh for summation
        mesh_energies: array(nk, nbnd)
            eigen-energies calculated for kmesh.kpoints
        mesh_component: array(nk, norb, nbnd)
            the orbital character
        dos_weight: float
            number of electrons occupying the orbital
        component_names: list[str]
            the name of the orbital components
        """
        nk, self._nbnd = mesh_energies.shape
        if nk != kmesh.numk:
            raise "run energies with kmesh!"

        self._kmesh = kmesh
        self._mesh_energies = mesh_energies.T.copy()
        self._dos_weight = dos_weight
        self._orbital_names = orbital_names
        self._mesh_components = mesh_components
        

    def calculate_projdos(self, dos_energies: np.ndarray) -> ProjDosResult:
        """calculate dos for the input array of energies
        
        result in unit 1/eV
        """
        nenergy = len(dos_energies)
        norb = self._mesh_components.shape[1]
        result = np.zeros((nenergy, norb))
        
        for ibnd in range(self._nbnd):
            if (np.min(self._mesh_energies[ibnd]) > np.max(dos_energies) or 
                np.max(self._mesh_energies[ibnd]) < np.min(dos_energies)):
                continue
            result += tools.cython_projdos_contribution_multiple(
                self._mesh_energies[ibnd], self._mesh_components[:,:,ibnd], 
                self._kmesh.tetras, dos_energies)
            
        return ProjDosResult(dos_energies, self._orbital_names, result*self._dos_weight)



def test_custom_argsort():
    """testing argsort for cython, but not used"""
    import numpy as np
    def argsort(e4):

        if e4[0] < e4[1]:
            part1_min = 0; part1_max = 1
        else:
            part1_min = 1; part1_max = 0
        if e4[2] < e4[3]:
            part2_min = 2; part2_max = 3
        else:
            part2_min = 3; part2_max = 2

        if e4[part1_min] < e4[part2_min]:
            ie1 = part1_min
            left1 = part2_min
        else:
            ie1 = part2_min
            left1 = part1_min
                
        if e4[part1_max] > e4[part2_max]:
            ie4 = part1_max
            left2 = part2_max
        else:
            ie4 = part2_max
            left2 = part1_max

        if e4[left1] < e4[left2]:
            ie2 = left1
            ie3 = left2
        else:
            ie2 = left2
            ie3 = left1

        return ie1, ie2, ie3, ie4
    
    for _ in range(1000):
        test_data = np.random.choice(1000, 4)
        i1,i2,i3,i4 = argsort(test_data)
        assert test_data[i1] <= test_data[i2]
        assert test_data[i2] <= test_data[i3]
        assert test_data[i3] <= test_data[i4]
