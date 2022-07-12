import numpy as np
import typing, copy

class Kmesh:
    def __init__(self, reciprocal_lattice: np.ndarray, nk: typing.List[int]):
        assert len(nk) == 3
        self._cell = np.array(reciprocal_lattice, dtype = float)
        self._nks = np.array(nk, dtype = int)
        self._kpoints = self._make_mesh(self._nks)

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
    
    @property
    def nks(self) -> typing.Tuple[int]:
        return self._nks[0], self._nks[1], self._nks[2]

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def numk(self) -> int:
        return self._nks[0] * self._nks[1] * self._nks[2]


class TetraKmesh(Kmesh):
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
        super().__init__(reciprocal_lattice, nk)
        self._tetras = self._make_tetrahedron()


    @property
    def tetras(self) -> np.ndarray:
        return self._tetras

    
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

