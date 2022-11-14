import typing, dataclasses
import numpy as np
from ...tools import find_RCL

__all__ = ["Kpath", "Kline"]

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
        parts = input_str.split()
        if len(parts) != 8:
            print(f"cannot format in to Kline: {input_str}")
        start_k = np.array(parts[1:4], dtype = float)
        end_k = np.array(parts[5:8], dtype = float)
        return cls(
            parts[0], start_k, parts[4], end_k
        )

    @property
    def kpoints(self) -> np.ndarray:
        return np.linspace(self.start_k, self.end_k, self.nk+1)

    def get_distance_given_cell(self, rlattice: np.ndarray) -> float:
        start_cart = rlattice.T.dot(self.start_k)
        end_cart = rlattice.T.dot(self.end_k)
        return np.linalg.norm(start_cart - end_cart)


@dataclasses.dataclass
class BandPathTick:
    symbol: int
    xpos: float

    def __eq__(self, other) -> bool:
        return self.symbol == other.symbol and np.isclose(self.xpos, other.xpos, atol=1e-4)


class Kpath:
    """
    this class provide the set of kpoints in a typical bandstructure plot.
    It needs: 
    - reciprocal_lattice: only used to determine the distance between k points in the plot
    - klines: specify the high symmetry points
    - quality: int in [-1, 0, 1], total number of kpoints = 150 + quality * 30
    It supplies:
    - ticks: a list of namedtuples giving position on the x axis and symbol
    - kpoints: a list of kpoints
    - xpos: the corresponding x positions of the kpoints
    """
    ntot_default = 150
    def __init__(self, 
        reciprocal_lattice: np.ndarray, klines : typing.List[Kline], quality: int = 0
    ) -> None:
        self._klines = klines
        self._latticeT = reciprocal_lattice.T
        self._kpoints = []
        self._xpos = []
        self._tics: typing.List[typing.Tuple[str, float]] = []

        _ktotal = self.ntot_default + quality * 30
        _distances = [ kl.get_distance_given_cell(reciprocal_lattice) for kl in klines ]
        _total_distance = sum(_distances)
        nks = [ round(_ktotal * (d / _total_distance)) for d in _distances ]
        
        start = 0
        for kl, nk in zip(klines, nks):
            cartesian_delta = self._latticeT.dot(kl.end_k - kl.start_k)
            cartesian_distance = np.linalg.norm(cartesian_delta)
            end = start + cartesian_distance
            dxs = np.linspace(start, end, nk + 1)

            self._xpos.append(dxs)
            start_tick = BandPathTick(kl.start_symbol, start)
            if start_tick not in self._tics: self._tics.append(start_tick)
            self._tics.append(BandPathTick(kl.end_symbol, end))
            self._kpoints.append(np.linspace(kl.start_k, kl.end_k, nk+1))

            start = end
        self._kpoints = np.vstack(self._kpoints)
        self._xpos = np.hstack(self._xpos)
    
    @classmethod
    def from_cell_string(
        cls, cell_realspace: np.ndarray, kline_strings: typing.List[str], quality: int = 0
    ) -> "Kpath":
        """
        from wannier/alamode style band path, for example:
        G 0.0 0.0 0.0 X 0.5 0.0 0.0
        """
        rcl = find_RCL(cell_realspace)
        klines = [ Kline.from_str(ks) for ks in kline_strings ]
        return cls(rcl, klines, quality)

    @property
    def ticks(self) -> typing.List[BandPathTick]:
        return self._tics

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def xpos(self) -> np.ndarray:
        return self._xpos

