import typing, dataclasses
import numpy as np
from .bandstructure import BandPathTick, Kpath
from ..tightbinding import TightBindingBase
from ..printing import get_orbital_symbol_from_lm
from ase.data import chemical_symbols

@dataclasses.dataclass
class OrbitName:
    chemical_symbol: str
    primitive_index: int
    l: int
    m: int

    @property
    def orbital_symbol(self) -> str:
        return get_orbital_symbol_from_lm(self.l, self.m)

    # eg: "Cl(2) px", "Fe(1) dxy", index (i) start from 1, giving the primitive cell index
    def __repr__(self) -> str:
        return f"{self.chemical_symbol}({self.primitive_index + 1}) {self.orbital_symbol}"
    

@dataclasses.dataclass
class FlatBandResult:
    x: np.ndarray 
    E: np.ndarray  # [nx, nbnd]
    c2: np.ndarray # [nx, nbnd, norb], squared coefficients, 
    orbnames: typing.List[str]
    ticks: typing.List[BandPathTick]

    def plot_data(self, filename: str, orbitals: typing.List[str]) -> None:
        assert set(orbitals) <= set(self.orbnames)

        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        axes.set_xlabel("High Symmetry Point")
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



def get_fatband_result(
    tb: TightBindingBase, kpath: Kpath
) -> FlatBandResult:
    energies, coefficients = tb.solveE_at_ks(kpath.kpoints)
    orbnames = []
    for basis in tb.basis:
        l = basis.l
        m = basis.m
        primitive_index = basis.pindex
        sym = chemical_symbols[tb.types[primitive_index]]
        orbnames.append(
            str(OrbitName(sym, primitive_index, l, m))
        )

    return FlatBandResult(
        kpath.xpos, energies, coefficients, orbnames, kpath.ticks
    )