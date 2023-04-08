import dataclasses

import numpy as np
from pymatgen.electronic_structure import bandstructure as pymgbs

from automaticTB.properties import tightbinding
from automaticTB.properties import reciprocal


@dataclasses.dataclass
class FermiSurfaceData:
    """a wrapper for fermi surface"""
    band: pymgbs.BandStructure

    @classmethod
    def from_tb_kmesh_singlespin(cls, 
        tb: tightbinding.TightBindingModel, 
        kmesh: reciprocal.Kmesh, 
        use_ibz: bool = True
    ) -> "FermiSurfaceData":
        if use_ibz:
            from automaticTB.properties import calc_mesh
            e, _ = calc_mesh.calculate_e_v_using_ibz(
                tb, kmesh.kpoints, kmesh.nks, energy_only=True)
        else:
            e, _ = tb.solveE_at_ks(kmesh.kpoints)

        bs = get_bandstructure_spindegenerate(
            tb.cell, tb.positions, tb.types, kmesh.kpoints, e
        )

        return cls(bs)

    def plot_fermi_surface(self, efermi: float) -> pymgbs.BandStructure:
        """plot fermi surface, show it and save to a html file"""
        from ifermi.surface import FermiSurface
        from ifermi.plot import FermiSurfacePlotter, show_plot

        fs = FermiSurface.from_band_structure(
            self.band, mu=efermi, wigner_seitz=True, calculate_dimensionality=True
            )

        fs_plotter = FermiSurfacePlotter(fs)
        plot = fs_plotter.get_plot()
        show_plot(plot)  # displays an interactive plot


def get_bandstructure_spindegenerate(
    cell: np.ndarray, 
    positions: np.ndarray, 
    types: np.ndarray, 
    kpoints: np.ndarray, 
    energies: np.ndarray, # nk, nbnd
) -> pymgbs.BandStructure:
    """adaptor function to give a Bandstructure object
    
    For spin generate case only, currently
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice
    from pymatgen.electronic_structure.core import Spin
    
    rlv = reciprocal.find_RCL(cell)

    l_new = Lattice(rlv)
    structure = Structure(cell, types, positions)

    assert len(kpoints) == len(energies)
    bs = pymgbs.BandStructure(
        kpoints, 
        {Spin.up: energies.T}, 
        l_new,
        efermi = 0.0,
        structure=structure
    )

    return bs
