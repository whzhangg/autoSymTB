import typing
import dataclasses

import numpy as np

from automaticTB import tools
from automaticTB.properties import kpoints
from automaticTB.properties import tightbinding


@dataclasses.dataclass
class BandStructureResult:
    x: np.ndarray
    E: np.ndarray # [nk, nbnd]
    ticks: typing.List[typing.Tuple[str, float]]


    def write_data_to_file(self, filename: str):
        if len(self.x) != self.E.shape[0]:
            raise "Size of energy not equal to x positions"
        ticks = [ "{:>s}:{:>10.6f}".format(symbol, xpos) for symbol, xpos in self.ticks]
        results = "# " + " & ".join(ticks) + "\n"
        results += f"# nk   {self.E.shape[0]}\n"
        results += f"# nbnd {self.E.shape[1]}\n"
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
    def from_tightbinding_and_kpath(
        cls, tb: tightbinding.TightBindingModel, kpath: kpoints.Kpath
    ) -> "BandStructureResult":
        energies = tb.solveE_at_ks(kpath.kpoints)
        return cls(
            kpath.xpos, energies, kpath.ticks
        )


    @classmethod
    def from_datafile(cls, filename: str):
        with open(filename, 'r') as f:
            aline = f.readline()
            parts = aline.lstrip("#").split("&")
            ticks = [
                (part.split(":")[0].strip(), float(part.split(":")[1])) for part in parts
            ]
            nk = int(f.readline().lstrip("#").split()[-1])
            nbnd = int(f.readline().lstrip("#").split()[-1])
            xy = []
            for _ in range(nk):
                xy.append( 
                    np.array(f.readline().split(), dtype=float) 
                )
            xy = np.vstack(xy)
        return cls(
            xy[:,0],
            xy[:,1:],
            ticks
        )


    def plot_data(
        self, filename: str, 
        yminymax: typing.Optional[typing.Tuple[float]] = None
    ):
        # plot simple band structure,
        # if scalar is given, scatter plot with point size
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        if yminymax is None:
            ymin = np.min(self.E)
            ymax = np.max(self.E)
            ymin = ymin - (ymax - ymin) * 0.05
            ymax = ymax + (ymax - ymin) * 0.05
        else:
            ymin, ymax = yminymax

        axes.set_xlim(min(self.x), max(self.x))
        axes.set_ylim(ymin, ymax)

        for ibnd in range(self.E.shape[1]):
            axes.plot(self.x, self.E[:, ibnd])
        for _, x in self.ticks:
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))

        fig.savefig(filename)
        
    def plot_fatband(
        self, filename: str,
        coefficients: typing.Dict[str, np.ndarray],
        yminymax: typing.Optional[typing.Tuple[float]] = None
    ) -> None:
        "fat band plot"
        nk, nbnd = self.E.shape
        ncoefficient = len(coefficients)
        for k,c in coefficients.items():
            if c.shape != self.E.shape:
                print(f"coefficient {k} has a wrong shape")

        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        if yminymax is None:
            ymin = np.min(self.E)
            ymax = np.max(self.E)
            ymin = ymin - (ymax - ymin) * 0.05
            ymax = ymax + (ymax - ymin) * 0.05
        else:
            ymin, ymax = yminymax

        axes.set_xlim(min(self.x), max(self.x))
        axes.set_ylim(ymin, ymax)

        nstates = int(nk * nbnd)
        state_x = np.zeros(nstates, dtype = float)
        state_y = np.zeros(nstates, dtype = float)
        #state_cs = np.zeros((n_orbgroup, nstate), dtype = float)
        state_c_dict = { key:np.zeros(nstates) for key in coefficients.keys() }

        count = 0
        for ix in range(nk):
            for ibnd in range(nbnd):
                state_x[count] = self.x[ix]
                state_y[count] = self.E[ix, ibnd]
                for key, c in coefficients.items():
                    state_c_dict[key][count] = c[ix, ibnd]
                count += 1

        for key, c in state_c_dict.items():
            axes.scatter(
                state_x, state_y, 
                s = c * 5, 
                marker = 'o', label = f"{key}")

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))
        axes.legend()
        fig.savefig(filename)


@dataclasses.dataclass
class OrbitName:
    chemical_symbol: str
    primitive_index: int
    l: int
    m: int

    @property
    def orbital_symbol(self) -> str:
        return tools.get_orbital_symbol_from_lm(self.l, self.m)

    # eg: "Cl(2) px", "Fe(1) dxy", index (i) start from 1, giving the primitive cell index
    def __repr__(self) -> str:
        return f"{self.chemical_symbol}({self.primitive_index + 1}) {self.orbital_symbol}"
    

@dataclasses.dataclass
class FatBandResult:
    x: np.ndarray 
    E: np.ndarray  # [nx, nbnd]
    c2: np.ndarray # [nx, nbnd, norb], squared coefficients, 
    orbnames: typing.List[str]
    ticks: typing.List[typing.Tuple[str, float]]

    def plot_fatband(
            self, filename: str, orbitalgroups: typing.Dict[str, typing.List[str]]) -> None:
        """
        orbitalgroups are dictionaries like, 
        where Symbol(primitive index) orbital_symbol give the orbital name
        {
            "Pb s": ["Pb(1) s"],
            "Cl p": ["Cl(2) px","Cl(2) py","Cl(2) pz"]
        }
        """
        # the following 4 lines check if the names are correct
        orbitals = []
        for orbitalgroup in orbitalgroups.values():
            orbitals += orbitalgroup
        assert set(orbitals) <= set(self.orbnames), self.orbnames

        n_orbgroup = len(orbitalgroups.keys())
        orbitalgroup_indices = {}
        for key, value in orbitalgroups.items():
            orbitalgroup_indices[key] = [
                [ self.orbnames.index(o) for o in value ]
            ]

        nx, nbnd = self.E.shape
        nstate = int(nx * nbnd)
        state_x = np.zeros(nstate, dtype = float)
        state_y = np.zeros(nstate, dtype = float)
        #state_cs = np.zeros((n_orbgroup, nstate), dtype = float)
        state_c_dict = { key:np.zeros(nstate) for key in orbitalgroups.keys() }

        count = 0
        for ix in range(nx):
            for ibnd in range(nbnd):
                state_x[count] = self.x[ix]
                state_y[count] = self.E[ix, ibnd]
                for key, group_indices in orbitalgroup_indices.items():
                    state_c_dict[key][count] = np.sum(self.c2[ix, ibnd, group_indices])
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

        for key, state_c in state_c_dict.items():
            axes.scatter(state_x, state_y, s = state_c * 5, marker = 'o', label = f"{key}")
        
        for _,x in self.ticks:
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))
        axes.legend()
        fig.savefig(filename)


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
        
        for _,x in self.ticks:
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))

        fig.savefig(filename)


    @classmethod
    def from_tightbinding_and_kpath(cls,
        tb: tightbinding.TightBindingModel, kpath: kpoints.Kpath
    ) -> "FatBandResult":
        energies, coefficients = tb.solveE_c2_at_ks(kpath.kpoints)
        orbnames = []
        for basis in tb.basis:
            l = basis.l
            m = basis.m
            primitive_index = basis.pindex
            sym = tools.chemical_symbols[tb.types[primitive_index]]
            orbnames.append(
                str(OrbitName(sym, primitive_index, l, m))
            )

        return cls(
            kpath.xpos, energies, coefficients, orbnames, kpath.ticks
        )

