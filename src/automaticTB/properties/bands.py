import typing
import dataclasses

import numpy as np

from automaticTB.properties import tightbinding
from automaticTB.properties import reciprocal


def interpolate_band(
    x: np.ndarray, 
    y: np.ndarray, 
    coefficients: typing.Dict[str, np.ndarray]
) -> typing.Tuple[np.ndarray, np.ndarray, typing.Dict[str, np.ndarray]]:
    """interpolate bands, use with care
    
    Parameters
    ----------
    x: array(nk)
        x values
    y: array(nk, nbnd)
        band energy
    coefficients: dict[str, array(nk, nbnd)]
        coefficients the same shape as y

    Outputs
    -------
    same as inputs

    """
    def _interpolate_single(x: np.ndarray):
        inter = 0.5 * (x[1:] + x[:-1])
        if x.ndim == 1:
            out = np.zeros(len(x) + len(inter))
            for i in range(len(inter)):
                out[2 * i] = x[i]
                out[2 * i + 1] = inter[i]
            out[-1] = x[-1]
        else:
            _, nbnd = x.shape
            out = np.zeros((len(x) + len(inter), nbnd))
            for i in range(len(inter)):
                out[2 * i, :] = x[i, :]
                out[2 * i + 1, :] = inter[i, :]
            out[-1,:] = x[-1,:]
        return out
    
    inter_v = {}
    for k, v in coefficients.items():
        inter_v[k] = _interpolate_single(v)
        
    return (
        _interpolate_single(x),
        _interpolate_single(y),
        inter_v
    )


def convert_to_scatter_point(
    x: np.ndarray, 
    y: np.ndarray, 
    coeff_dict: typing.Tuple[str, np.ndarray]
) -> typing.Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """convert band to scattered points
    
    Parameters
    ----------
    x: array(nk)
        x position
    y: array(nk, nbnd)
        energies
    coeff_dict: dict[str, array(nk, nbnd)]
        coefficients for band states with key

    """
    nk, nbnd = y.shape
    nstate = int(nk * nbnd)
    flat_x = np.zeros(nstate)
    flat_y = np.zeros(nstate)
    flat_dict = {k: np.zeros(nstate) for k in coeff_dict.keys()}
    count = 0
    for ik in range(nk):
        for ibnd in range(nbnd):
            flat_x[count] = x[ik]
            flat_y[count] = y[ik, ibnd]
            for k, c in coeff_dict.items():
                flat_dict[k][count] = c[ik, ibnd]
            count += 1
    return flat_x, flat_y, flat_dict


@dataclasses.dataclass
class BandStructure:
    x: np.ndarray 
    e: np.ndarray  # [nx, nbnd]
    c: np.ndarray  # [nx, norb, nbnd] 
    orbnames: typing.List[str]
    ticks: typing.List[typing.Tuple[str, float]]

    def plot_fatband(
        self, 
        filename: str, 
        orbitalgroups: typing.Dict[str, typing.List[str]],
        take_norm: bool = True,
        interpolate: bool = False,
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

        orbitalgroup_coefficients = {}
        for k, vs in orbitalgroups.items():
            coef = np.zeros_like(self.e)
            for v in vs:
                iv = self.orbnames.index(v)
                if take_norm:
                    coef += np.abs(self.c[:, iv, :])**2
                else:
                    coef += self.c[:, iv, :]
            orbitalgroup_coefficients[k] = coef

        if interpolate:
            bandx, bandy, orbcoe = interpolate_band(self.x, self.e, orbitalgroup_coefficients)
        else:
            bandx, bandy, orbcoe = self.x, self.e, orbitalgroup_coefficients

        state_x, state_y, state_c_dict = convert_to_scatter_point(
            bandx, bandy, orbcoe)

        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        #axes = fig.subplots()
        axes = fig.add_axes((0.1, 0.15, 0.6, 0.7))

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        if yminymax is not None:
            ymin, ymax = yminymax
        else:
            ymin = np.min(bandy)
            ymax = np.max(bandy)
            ymin = ymin - (ymax - ymin) * 0.05
            ymax = ymax + (ymax - ymin) * 0.05

        axes.set_xlim(np.min(bandx), np.max(bandx))
        axes.set_ylim(ymin, ymax)

        for key, state_c in state_c_dict.items():
            axes.scatter(state_x, state_y, s = state_c * 5, marker = 'o', label = f"{key}")
        
        for _,x in self.ticks:
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))
        #axes.legend()
        fig.legend(frameon = False, loc = (0.75, 0.15))
        fig.savefig(filename)


    def plot_band(
        self, 
        filename: str, 
        yminymax = None,
        interpolate: bool = False
    ) -> None:
        # plot simple band structure
        if interpolate:
            bandx, bandy, _ = interpolate_band(self.x, self.e, {})
        else:
            bandx, bandy = self.x, self.e

        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        #axes.set_xlabel("High Symmetry Point")
        
        if yminymax is not None:
            ymin, ymax = yminymax
        else:
            ymin = np.min(bandy)
            ymax = np.max(bandy)
            ymin = ymin - (ymax - ymin) * 0.05
            ymax = ymax + (ymax - ymin) * 0.05

        axes.set_xlim(min(bandx), max(bandx))
        axes.set_ylim(ymin, ymax)
        
        for _,x in self.ticks:
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ x for _,x in self.ticks ]
        tick_s = [ s for s,_ in self.ticks ]
        
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))

        for ibnd in range(self.e.shape[1]):
            axes.plot(bandx, bandy[:, ibnd])

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
                # not certain if it really help to find band order...
                # how to check? maybe by perturbed band
                bandorder_reference = bandorder_reference * 0.1 + c[:, sort_indices] * 0.9
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