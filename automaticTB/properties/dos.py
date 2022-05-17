import numpy as np
import abc, typing, copy
from ..utilities import find_RCL
from ..hamiltionian.model import TightBindingBase

class SpinDegenerateDOSBase(abc.ABC):
    """for all the three public methods, should return dict containing dos of each spin"""

    @property
    @abc.abstractclassmethod
    def x(self) -> np.ndarray:
        raise

    @property
    @abc.abstractclassmethod
    def dos(self) -> np.ndarray:
        """this method should give the density of states of the whole materials """
        raise

    @abc.abstractclassmethod
    def single_dos(self, e: float) -> np.ndarray:
        """this method should give density of states at a specific energy"""
        raise

    @abc.abstractclassmethod
    def nsum(self, e: float) -> np.ndarray:
        """this method should give the electron counts """
        raise

    def write_data_to_file(self, filename: str):
        assert self.x.shape == self.dos.shape
        with open(filename, 'w') as f:
            for x, y in zip(self.x, self.dos):
                f.write("{:>20.12e}{:>20.12e}\n".format(x,y))

    def plot_data(self, filename: str):
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


class Kmesh:
    # kmesh designed for tetrahedron methods, which generate a set of kpoints as well as tetrahedrons
    MESH_SHIFT = ( (0, 0, 0),   # 1 0
               (1, 0, 0),   # 2 1
               (0, 1, 0),   # 3 2
               (1, 1, 0),   # 4 3
               (0, 0, 1),   # 5 4
               (1, 0, 1),   # 6 5
               (0, 1, 1),   # 7 6
               (1, 1, 1) )  # 8 7

    def __init__(self, reciprocal_lattice: np.ndarray, nk: typing.List[int]):
        assert len(nk) == 3
        self._cell = np.array(reciprocal_lattice, dtype = float)
        self._nks = np.array(nk, dtype = int)
        self._kpoints = self._make_mesh(self._nks)

        self._tetras = self._make_tetrahedron()

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def numk(self) -> int:
        return self._nks[0] * self._nks[1] * self._nks[2]

    @property
    def tetras(self) -> np.ndarray:
        return self._tetras

    def _make_mesh(self, nks: typing.List[int]) -> np.ndarray:
        totk = nks[0] * nks[1] * nks[2]
        xk = np.zeros((totk,3))
        for i in range(nks[0]):
            for j in range(nks[1]):
                for k in range(nks[2]):
                    ik = k + j * nks[2] + i * nks[1] * nks[2]
                    xk[ik,0] = i / nks[0]
                    xk[ik,1] = j / nks[1]
                    xk[ik,2] = k / nks[2]
        return xk

    def _get_knum(self, xk: np.ndarray) -> int:
        i = np.rint(xk[0]*self._nks[0] + 2 * self._nks[0]) % self._nks[0] 
        j = np.rint(xk[1]*self._nks[1] + 2 * self._nks[1]) % self._nks[1] 
        k = np.rint(xk[2]*self._nks[2] + 2 * self._nks[2]) % self._nks[2] 
        return int( k + j * self._nks[2] + i * self._nks[1] * self._nks[2] )

    def _make_tetrahedron(self) -> np.ndarray:
        # determine how to shuffle the index
        mesh_shape = copy.deepcopy(self._cell)
        for i in range(3):
            mesh_shape[i,:] /= self._nks[i]
        G_order, G_neg = self._OptimizeGs(mesh_shape)
        G_order_rev = np.argsort(G_order)

        tetra = np.zeros((self.numk * 6, 4), dtype = int)
        subcell_tetras = [(0, 1, 2, 5), (0, 2, 4, 5), (2, 4, 5, 6), 
                          (2, 5, 6, 7), (2, 3, 5, 7), (1, 2, 3, 5)]

        mesh_shift_np = np.array(self.MESH_SHIFT, dtype = int)
        mesh_shift_permuted = np.zeros(mesh_shift_np.shape, dtype = int)
        for i in range(len(mesh_shift_np)):
            mesh_shift_permuted[i] = G_neg * ( mesh_shift_np[i,G_order_rev] )

        for ik, xk in enumerate(self.kpoints):
            for it, corner in enumerate(subcell_tetras):
                c = []
                for ic in range(4):
                    xc = xk + mesh_shift_permuted[corner[ic]] * 1.0 / self._nks
                    c.append(self._get_knum(xc))
                tetra[ik * 6 + it,:] = c
        return tetra

    def _OptimizeGs(self, v: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        '''
        this is taken from tetra/ksample.py, which gives the shuffle needed to 
        make 3,6 the cloest
        '''
        permutations = ( (0, 1, 2), (0, 2, 1), (1, 0, 2), 
                        (1, 2, 0), (2, 0, 1), (2, 1, 0) )
        signs = ( ( 1, 1, 1), ( 1, 1, -1), ( 1, -1, 1), ( 1, -1, -1),
                (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1) )
        G_order = None
        G_neg = None
        opt_36 = float("inf")
        for perm, sign in ( (p,s) for p in permutations for s in signs ):
            transformed_v = np.diag(sign).dot(v)[perm,:]
            k3_to_k6 = np.linalg.norm( 
                transformed_v.T.dot(self.MESH_SHIFT[2]) - transformed_v.T.dot(self.MESH_SHIFT[5]) 
            )
            if k3_to_k6 < opt_36:
                opt_36 = k3_to_k6
                G_order = perm
                G_neg = sign
        return np.array(G_order, dtype = int), np.array(G_neg, dtype = int) 

    def _distance(self, k1: np.ndarray, k2: np.ndarray) -> float:
        return np.linalg.norm(k1 - k2)


class TetraDOS(SpinDegenerateDOSBase):
    r"""
    This tetrahedron method is adopted from https://github.com/tflovorn/tetra
    Theory:
        k point density in the reciprocal space is given by: V / (8 * pi)^3
        where V is the volume of the crystal: V = N * V_cell

        The density of state of the whole system is given by:
            G(E) = weight * [ V / (4 * pi^2 ) ( 2m* / hbar )^{3/2} sqrt(E) ]
        where weight = 1, 2 the spin degeneracy

        The density of state per unit cell, which is calculated by the dos() method 
        provided here, is:
            g(E) = G(E) / N
                = weight * [ V_cell / (4 * pi^2 ) ( 2m* / hbar )^{3/2} sqrt(E) ]

        nsum is the summation of number of states in the BZ, per unit cell below a 
        given energy.
        nsum is related to the dos calculated here (dos()) by:
            n = int_{-\infty}^{E} g(E') dE'
        it is given by:
            n = weight * [ 1/N \sum_{k; Ek < E} ] 
        which should equal to the number of electrons per unit cell.
    """

    def __init__(self, 
        kmesh: Kmesh, 
        mesh_energies: np.ndarray, 
        dos_energies: np.ndarray
    ):
        nk, self._nbnd = mesh_energies.shape
        if nk != kmesh.numk:
            raise "run energies with kmesh!"

        self._kmesh = kmesh
        self._mesh_energies = mesh_energies
        self._dos_weight = 2.0 # spin nondegenerate
        self._dos_energies = dos_energies
        self._result = {}

    @property
    def x(self) -> np.ndarray:
        return self._dos_energies

    @property
    def dos(self) -> np.ndarray:
        if "dos" not in self._result:
            result = np.zeros_like(self._dos_energies)
            for ie, e in enumerate(self._dos_energies):
                result[ie] = self.single_dos(e)

            self._result["dos"] = result

        return self._result["dos"]

    def nsum(self, e: float) -> dict:
        # sum the number of states with energy below E
        nsum = 0.0

        for ibnd in range(self._nbnd):
            nsum += self._sum_contribution(e, ibnd)
            
        nsum *= self._dos_weight
        return nsum

    def single_dos(self, e: float) -> np.ndarray:
        # summed contribution from all bands at energy e
        dos = 0.0

        for ibnd in range(self._nbnd):
            dos += self._dos_contribution(e, ibnd) 

        dos *= self._dos_weight
        return dos

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


def get_tetrados_result(
    tb: TightBindingBase, ngrid: typing.Tuple[int, int, int], x_density: int = 50, 
) -> TetraDOS:
    reciprocal_cell = find_RCL(tb.cell)

    kmesh = Kmesh(reciprocal_cell, ngrid)
    energies = tb.solveE_at_ks(kmesh.kpoints)

    e_min = np.min(energies)
    e_max = np.max(energies)
    e_min = e_min - (e_max - e_min) * 0.05
    e_max = e_max + (e_max - e_min) * 0.05
    
    return TetraDOS(kmesh, energies, np.linspace(e_min, e_max, x_density))