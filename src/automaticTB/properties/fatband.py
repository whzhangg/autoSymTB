import typing, dataclasses
import numpy as np
from .bandstructure import BandPathTick, Kpath
from ..tightbinding import TightBindingBase
from ..printing import get_orbital_symbol_from_lm
from ase.data import chemical_symbols

# TODO 

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

    def plot_data(self, filename: str, orbitalgroups: typing.List[typing.List[str]]) -> None:
        """
        Basically, we are plotting points in a fat band plot, so we are free to flatten nx and nbnd
        """
        orbitals = []
        for orbitalgroup in orbitalgroups:
            orbitals += orbitalgroup
        assert set(orbitals) <= set(self.orbnames), self.orbnames

        n_orbgroup = len(orbitalgroups)
        orbitalgroup_indices = []
        for orbitalgroup in orbitalgroups:
            orbitalgroup_indices.append(
                [ self.orbnames.index(o) for o in orbitalgroup ]
            )

        nx, nbnd = self.E.shape
        nstate = int(nx * nbnd)
        state_x = np.zeros(nstate, dtype = float)
        state_y = np.zeros(nstate, dtype = float)
        state_cs = np.zeros((n_orbgroup, nstate), dtype = float)

        count = 0
        for ix in range(nx):
            for ibnd in range(nbnd):
                state_x[count] = self.x[ix]
                state_y[count] = self.E[ix, ibnd]
                for igroup, group_indices in enumerate(orbitalgroup_indices):
                    state_cs[igroup, count] = np.sum(self.c2[ix, ibnd, group_indices])
                count += 1

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

        for igroup, state_c in enumerate(state_cs):
            axes.scatter(state_x, state_y, s = state_c * 5, marker = 'o', label = f"Group {igroup+1}")
        
        for tick in self.ticks:
            x = tick.xpos
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ tick.xpos for tick in self.ticks ]
        tick_s = [ tick.symbol for tick in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))
        axes.legend()
        fig.savefig(filename)



def get_fatband_result(
    tb: TightBindingBase, kpath: Kpath
) -> FlatBandResult:
    energies, coefficients = tb.solveE_c2_at_ks(kpath.kpoints)
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