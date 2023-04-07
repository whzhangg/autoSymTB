import typing
import dataclasses

import numpy as np

from automaticTB.properties import tightbinding
from automaticTB.properties import reciprocal


@dataclasses.dataclass
class BandStructure:
    x: np.ndarray 
    E: np.ndarray  # [nx, nbnd]
    c: np.ndarray  # [nx, norb, nbnd] 
    orbnames: typing.List[str]
    ticks: typing.List[typing.Tuple[str, float]]

    def plot_fatband(
        self, 
        filename: str, 
        orbitalgroups: typing.Dict[str, typing.List[str]],
        yminymax: typing.Optional[typing.Tuple[float,float]] = None
    ) -> None:
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

        coefficient = np.abs(self.c)**2

        orbitalgroup_indices = {}
        for key, value in orbitalgroups.items():
            orbitalgroup_indices[key] = np.array([
                [ self.orbnames.index(o) for o in value ]
            ])

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
                    
                    state_c_dict[key][count] = np.sum(coefficient[ix, group_indices, ibnd])
                count += 1

        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        if yminymax is not None:
            ymin, ymax = yminymax
        else:
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


    def plot_band(self, filename: str, yminymax = None):
        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        
        if yminymax is not None:
            ymin, ymax = yminymax
        else:
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
        tb: tightbinding.TightBindingModel, kpath: reciprocal.Kpath, order_band: bool = True
    ) -> "BandStructure":
        nk = len(kpath.kpoints)
        nbnd = tb.nbasis
        
        ws, cs = tb.solveE_at_ks(kpath.kpoints)
        if order_band:
            eig = np.zeros((nk, nbnd), dtype=np.double)
            vec = np.zeros((nk, nbnd, nbnd), dtype = np.cdouble)
            bandorder_reference = cs[0]
            for ik in range(nk):
                w = ws[ik]
                c = cs[ik]
                sort_indices = _find_sort_indices(bandorder_reference, c)
                bandorder_reference = bandorder_reference * 0.4 + c[:, sort_indices] * 0.6
                vec[ik] = c[:,sort_indices]
                eig[ik] = w[sort_indices]
        else:
            eig, vec = ws, cs

        return cls(kpath.xpos, eig, vec, tb.basis_name, kpath.ticks)
        

def _find_sort_indices(c_ref: np.ndarray, c_in: np.ndarray) -> np.ndarray:
    """ find the order of eigenvalues by comparing the eigenvector

    `c_in` are the eigenvector solved and `c_ref` is the ref. 
    The ith eigenvector corresponding to ith eigenvalue is c[:,i]
    """
    nbasis = len(c_ref)
    found = []
    for i in range(nbasis):
        projections = np.abs(np.dot(c_ref[:,i].conjugate(), c_in))
        #print(np.round(projections,3))
        maximum = -1
        for j, p in enumerate(projections):
            if j in found: continue
            if p > maximum:
                which = j
                maximum = p
        found.append(which)
                
    return np.array(found)