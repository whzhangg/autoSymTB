import typing
import dataclasses

import numpy as np


def recommand_kgrid(rcell: np.ndarray, nk: int) -> typing.Tuple[int,int,int]:
    norms = np.linalg.norm(rcell, axis=1)
    ratio = nk / np.mean(norms)
    return np.array(norms * ratio, dtype=int)


def find_RCL(cell: np.ndarray) -> np.ndarray:
    """find the reciprocal lattice vectors"""
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]
    volumn=np.cross(a1,a2).dot(a3)
    b1=(2*np.pi/volumn)*np.cross(a2,a3)
    b2=(2*np.pi/volumn)*np.cross(a3,a1)
    b3=(2*np.pi/volumn)*np.cross(a1,a2)
    return np.vstack([b1,b2,b3])


def get_volume(cell: np.ndarray) -> float:
    """return volume in A^3"""
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]
    return np.cross(a1,a2).dot(a3)


@dataclasses.dataclass
class Kline:
    """
    kpoints will include the first and the last k point
    """
    start_symbol: str
    start_k: np.ndarray 
    end_symbol: str
    end_k : np.ndarray

    @classmethod
    def from_str(cls, input_str: str) -> "Kline":
        """construct a kpath section from a line of string
        
        For example: 'G 0.0000 0.0000 0.0000 K 0.3750 0.3750 0.7500'
        """
        parts = input_str.split()
        if len(parts) != 8:
            print(f"cannot format in to Kline: {input_str}")
        start_k = np.array(parts[1:4], dtype = float)
        end_k = np.array(parts[5:8], dtype = float)
        return cls(
            parts[0], start_k, parts[4], end_k)


    def kpoints(self, nk) -> np.ndarray:
        """return nk+1 points on this line, including both ends"""
        return np.linspace(self.start_k, self.end_k, nk+1)


    def get_distance_given_cell(self, rlattice: np.ndarray) -> float:
        """cartesian distance between two end point in reciprocal"""
        start_cart = rlattice.T.dot(self.start_k)
        end_cart = rlattice.T.dot(self.end_k)
        return np.linalg.norm(start_cart - end_cart)


class Kpath:
    """provide the set of kpoints, corresponding x and ticks for path

    It require reciprocal lattice, a set of klines.
    The number of points in the path can be controled as 
    200 + quality * 25
    """
    ntot_default = 150
    def __init__(self, 
        reciprocal_lattice: np.ndarray, klines : typing.List[Kline], quality: int
    ) -> None:
        self._kpoints = []
        self._xpos = []
        self._tics: typing.List[typing.Tuple[str, float]] = []

        latticeT = reciprocal_lattice.T
        _ktotal = self.ntot_default + quality * 30
        _distances = [ kl.get_distance_given_cell(reciprocal_lattice) for kl in klines ]
        _total_distance = sum(_distances)

        nks = [ round(_ktotal * (d / _total_distance)) for d in _distances ]
        
        start = 0
        for kl, nk in zip(klines, nks):
            cartesian_delta = latticeT.dot(kl.end_k - kl.start_k)
            cartesian_distance = np.linalg.norm(cartesian_delta)
            end = start + cartesian_distance
            dxs = np.linspace(start, end, nk + 1)

            self._xpos.append(dxs)

            if len(self._tics) == 0:
                self._tics.append((kl.start_symbol, start))
            else:
                last_symbol = self._tics[-1][0]
                last_xpos = self._tics[-1][1]
                if last_symbol == kl.start_symbol and np.isclose(last_xpos, start, atol=1e-4):
                    pass
                else:
                    self._tics.pop(-1)
                    self._tics.append((f"{last_symbol}|{kl.start_symbol}", start))
            
            self._tics.append((kl.end_symbol, end))
            self._kpoints.append(np.linspace(kl.start_k, kl.end_k, nk+1))

            start = end

        self._kpoints = np.vstack(self._kpoints)
        self._xpos = np.hstack(self._xpos)


    @classmethod
    def from_cell_pathstring(
        cls, cell: np.ndarray, kpath_string: typing.List[str], quality: int = 0
    ) -> "Kpath":
        kpaths = [Kline.from_str(s) for s in kpath_string]
        rcell = find_RCL(cell)
        return cls(rcell, kpaths, quality)


    @property
    def ticks(self) -> typing.List[typing.Tuple[str, float]]:
        return self._tics


    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints


    @property
    def xpos(self) -> np.ndarray:
        return self._xpos


class Kmesh:
    """Mesh of k points starting from the origin"""
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
    
    @classmethod
    def from_cell_nk(
        cls, cell: np.ndarray, nk: typing.Union[int, typing.Tuple[int,int,int]]
    ) -> "Kmesh":
        """get a mesh of k points
        
        Parameters
        ----------
        cell: np.ndarray(3,3)
            the lattice in real space
        nk: int or (int,int,int)
            the kgrid parameter
        
        """
        rcell = find_RCL(cell)
        if isinstance(nk, int):
            nks = recommand_kgrid(rcell, nk)
        else:
            nks = nk
        
        return cls(rcell, nks)
    
    @property
    def reciprocal_lattice(self) -> np.ndarray:
        return self._cell
    
    @property
    def nks(self) -> typing.Tuple[int]:
        return self._nks[0], self._nks[1], self._nks[2]

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def numk(self) -> int:
        return self._nks[0] * self._nks[1] * self._nks[2]
    

class _UnitCell:
    """a wrapper related to kpoints"""
    def __init__(self, cell: np.ndarray) -> None:
        self.cell = cell
        self.reciprocal_cell = find_RCL(cell)


    def get_Kmesh(self, nk: typing.Union[int, typing.Tuple[int, int, int]]) -> Kmesh:
        """get a mesh of k points

        a single number is enough to automatically generate suitable
        k points.
        """
        if isinstance(nk, int):
            nks = self.recommend_kgrid(nk)
        else:
            nks = nk
        return Kmesh(self.reciprocal_cell, nks)


    def recommend_kgrid(self, gridsize: int) -> typing.Tuple[int, int, int]:
        norms = np.linalg.norm(self.reciprocal_cell, axis=1)
        ratio = gridsize / np.mean(norms)
        return np.array(norms * ratio, dtype=int)


    def get_kpath_from_path_string(self, 
            kpath_string: typing.List[str], quality: int = 0) -> Kpath:
        """return a Kpath object with string in the style of wannier
        
        eg: ["X 0.5 0.0 0.0 G 0.0 0.0 0.0", "G 0.0 0.0..", ... ]
        """
        kpaths = [Kline.from_str(s) for s in kpath_string]
        return Kpath(self.reciprocal_cell, kpaths, quality)


    def get_volume(self) -> float:
        return get_volume(self.cell)


def _test_kpath():
    cubic_band = [
        "L   0.5   0.5  0.5  G  0.0   0.0  0.0",
        "G   0.0   0.0  0.0  X  0.5   0.0  0.5",
        "X   0.5   0.0  0.5  U  0.625 0.25 0.625",
        "K 0.375 0.375 0.75  G  0.0   0.0  0.0"
    ]
    klines = [Kline.from_str(p) for p in cubic_band]
    cell = np.array([[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]])
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]
    volumn=np.cross(a1,a2).dot(a3)
    b1=(2*np.pi/volumn)*np.cross(a2,a3)
    b2=(2*np.pi/volumn)*np.cross(a3,a1)
    b3=(2*np.pi/volumn)*np.cross(a1,a2)
    rcell = np.vstack([b1,b2,b3])
    path = Kpath(rcell, klines)

    ticks_answer = [
        ('L', 0), ('G', 2.7206990463513265), ('X', 5.862291699941119), 
        ('U', 6.973012434480711), ('U|K', 6.973012434480711), ('G', 10.305174638099485)]

    for tick, tickref in zip(path.ticks, ticks_answer):
        assert tick[0] == tickref[0]
        assert np.isclose(tick[1], tickref[1])
