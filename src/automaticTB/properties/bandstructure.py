import typing, dataclasses
import numpy as np
from ..tightbinding import TightBindingBase

@dataclasses.dataclass
class Kline:
    # kpoints will include the first and the last k point
    start_symbol: str
    start_k: np.ndarray 
    end_symbol: str
    end_k : np.ndarray
    nk: int

    @property
    def kpoints(self) -> np.ndarray:
        return np.linspace(self.start_k, self.end_k, self.nk+1)


@dataclasses.dataclass
class BandPathTick:
    symbol: int
    xpos: float

    def __eq__(self, other) -> bool:
        return self.symbol == other.symbol and np.isclose(self.xpos, other.xpos, atol=1e-4)


class Kpath:
    def __init__(self, reciprocal_lattice: np.ndarray, klines : typing.List[Kline]) -> None:
        self._klines = klines
        self._latticeT = reciprocal_lattice.T
        self._kpoints = []
        self._xpos = []
        self._tics: typing.List[typing.Tuple[str, float]] = []

        start = 0
        for kl in klines:
            cartesian_delta = self._latticeT.dot(kl.end_k - kl.start_k)
            cartesian_distance = np.linalg.norm(cartesian_delta)
            end = start + cartesian_distance
            dxs = np.linspace(start, end, kl.nk + 1)

            self._xpos.append(dxs)
            start_tick = BandPathTick(kl.start_symbol, start)
            if start_tick not in self._tics: self._tics.append(start_tick)
            self._tics.append(BandPathTick(kl.end_symbol, end))
            self._kpoints.append(kl.kpoints)

            start = end
        self._kpoints = np.vstack(self._kpoints)
        self._xpos = np.hstack(self._xpos)
        
    @property
    def ticks(self) -> typing.List[BandPathTick]:
        return self._tics

    @property
    def kpoints(self) -> np.ndarray:
        return self._kpoints

    @property
    def xpos(self) -> np.ndarray:
        return self._xpos


@dataclasses.dataclass
class BandStructureResult:
    x: np.ndarray
    E: np.ndarray
    ticks: typing.List[BandPathTick]

    def write_data_to_file(self, filename: str):
        if len(self.x) != self.E.shape[0]:
            raise "Size of energy not equal to x positions"
        ticks = [ "{:>s}:{:>10.6f}".format(tick.symbol, tick.xpos) for tick in self.ticks]
        results = " & ".join(ticks) + "\n"
        results += f"nk   {self.E.shape[0]}\n"
        results += f"nbnd {self.E.shape[1]}\n"
        dataformat = "{:>20.12e}"
        for x, ys in zip(self.x, self.E):
            results += dataformat.format(x)
            for y in ys:
                results += dataformat.format(y)
            results += "\n"
        with open(filename, 'w') as f:
            f.write(results)


    def __eq__(self, o: object) -> bool:
        return np.allclose(self.x, o.x) and \
               np.allclose(self.E, o.E) and \
                   self.ticks == o.ticks


    @classmethod
    def from_datafile(cls, filename: str):
        with open(filename, 'r') as f:
            aline = f.readline()
            parts = aline.split("&")
            ticks = [
                BandPathTick(part.split(":")[0].strip(), float(part.split(":")[1])) for part in parts
            ]
            nk = int(f.readline().split()[-1])
            nbnd = int(f.readline().split()[-1])
            xy = []
            for ik in range(nk):
                xy.append( 
                    np.array(f.readline().split(), dtype=float) 
                )
            xy = np.vstack(xy)
        return cls(
            xy[:,0],
            xy[:,1:],
            ticks
        )


    def plot_data(self, filename: str):
        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        ymin = np.min(self.E)
        ymax = np.max(self.E)
        ymin = ymin - (ymax - ymin) * 0.05
        ymax = ymax + (ymax - ymin) * 0.05

        axes.set_xlim(min(self.x), max(self.x))
        axes.set_ylim(ymin, ymax)

        for ibnd in range(self.E.shape[1]):
            axes.plot(self.x, self.E[:, ibnd])
        for tick in self.ticks:
            x = tick.xpos
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ tick.xpos for tick in self.ticks ]
        tick_s = [ tick.symbol for tick in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))

        fig.savefig(filename)


def get_bandstructure_result(
    tb: TightBindingBase, kpath: Kpath
) -> BandStructureResult:
    energies = tb.solveE_at_ks(kpath.kpoints)
    return BandStructureResult(
        kpath.xpos, energies, kpath.ticks
    )