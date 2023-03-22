import numpy as np
import joblib

from automaticTB import tools
from .tetra_mesh import TetraKmesh


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
    _dos_weight = 2.0
    parallel_threshold = 8000

    def __init__(self, 
        kmesh: TetraKmesh, 
        mesh_energies: np.ndarray, 
        dos_energies: np.ndarray
    ):
        nk, self._nbnd = mesh_energies.shape
        if nk != kmesh.numk:
            raise "run energies with kmesh!"

        self._kmesh = kmesh
        self._mesh_energies = mesh_energies.T.copy()
        self._dos_energies = dos_energies
        self._result = {}


    @property
    def x(self) -> np.ndarray:
        return self._dos_energies


    @property
    def dos(self) -> np.ndarray:
        if "dos" not in self._result:
            self._result["dos"] = self._calculate_dos()
        return self._result["dos"]


    @tools.timefn
    def _calculate_dos(self):
        result = np.zeros_like(self._dos_energies)
        #for ie, e in enumerate(self._dos_energies):
        #    result[ie] = self.single_dos(e)
        for ibnd in range(self._nbnd):
            if (np.min(self._mesh_energies[ibnd]) > np.max(self._dos_energies) or 
                np.max(self._mesh_energies[ibnd]) < np.min(self._dos_energies)):
                continue
            result += tools.cython_dos_contribution_multiple(
                self._mesh_energies[ibnd], self._kmesh.tetras, self._dos_energies)
        
        return result

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

    # not used
    def _dos_contribution(self, e: float, band_index: int) -> float:
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

    # not used
    def _sum_contribution(self, e: float, band_index: int) -> float:
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
