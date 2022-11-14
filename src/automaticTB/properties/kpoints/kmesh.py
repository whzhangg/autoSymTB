import numpy as np
import typing

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

