import typing
import copy

import numpy as np
from automaticTB.properties import reciprocal
from automaticTB import tools


class TetraKmesh(reciprocal.Kmesh):
    """kmesh designed for tetrahedron methods
    
    It generate a set of kpoints as well as tetrahedrons
    """
    MESH_SHIFT = ( 
        (0, 0, 0),   # 1 0
        (1, 0, 0),   # 2 1
        (0, 1, 0),   # 3 2
        (1, 1, 0),   # 4 3
        (0, 0, 1),   # 5 4
        (1, 0, 1),   # 6 5
        (0, 1, 1),   # 7 6
        (1, 1, 1)    # 8 7
    )

    def __init__(self, reciprocal_lattice: np.ndarray, nk: typing.List[int]):
        super().__init__(reciprocal_lattice, nk)
        self._tetras = self._make_tetrahedron()


    @property
    def tetras(self) -> np.ndarray:
        return self._tetras


    def _make_tetrahedron(self) -> np.ndarray:
        # determine how to shuffle the index
        mesh_shape = copy.deepcopy(self._cell)
        for i in range(3):
            mesh_shape[i,:] /= self._nks[i]
        G_order, G_neg = self._OptimizeGs(mesh_shape)
        G_order_rev = np.argsort(G_order)

        subcell_tetras = [(0, 1, 2, 5), (0, 2, 4, 5), (2, 4, 5, 6), 
                          (2, 5, 6, 7), (2, 3, 5, 7), (1, 2, 3, 5)]

        mesh_shift_np = np.array(self.MESH_SHIFT, dtype = int)
        mesh_shift_permuted = np.zeros(mesh_shift_np.shape, dtype = int)
        for i in range(len(mesh_shift_np)):
            mesh_shift_permuted[i] = G_neg * ( mesh_shift_np[i,G_order_rev] )
        
        nks = self._nks.copy()
        nks.dtype = np.intp
        return tools.cython_create_tetras(
            self.kpoints, 
            np.array(subcell_tetras, dtype = np.intp), 
            np.array(mesh_shift_permuted, dtype = np.intp), 
            nks[0], nks[1], nks[2]
        )


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

