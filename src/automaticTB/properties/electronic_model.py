import typing
import os

import numpy as np

from automaticTB import tools
from automaticTB.properties import dos
from automaticTB.properties import kpoints
from automaticTB.properties import tightbinding
from automaticTB.properties import bands
from automaticTB.properties import transport

class ElectronicModel:
    """Property wrapper

    This class is a main wrapper that extract properties from the
    tightbinding model.
    """
    def __init__(self, tbmodel: tightbinding.TightBindingModel) -> None:
        self.tb = tbmodel


    @property
    def cell(self) -> np.ndarray:
        return self.tb.cell


    @property
    def positions(self) -> np.ndarray:
        return self.tb.positions


    @property
    def types(self) -> np.ndarray:
        return self.tb.types


    @tools.timefn
    def get_transportproperties(
        self, t_range: np.ndarray, mu_range: np.ndarray, 
        valence: int, tau: float, nkgrid: int, weight: float = 2.0
    ) -> transport.TransportResult:
        
        calculated = transport.TransportResult.calculate_from_tightbinding(
            self.tb, t_range, mu_range, valence, tau, nkgrid, weight
        )
        return calculated


    def get_effectivemass(
            self, kpoint: np.ndarray, ibnd: int) -> transport.ParabolicBandFitting:
        """fit effective mass
        
        When k point of the band maximum is not clear, it's necessary
        to go along the kpoints and find the extreme. If the k point 
        used is not near the maximum, the energy surface found could
        fit to other parabolics instead of the parabolic of the 
        band edge. It's always necessary to check if the obtained 
        fitting kmax belong to the input energy surface.  
        """

        mass = transport.BandEffectiveMass(self.tb)
        return mass.fit_parabolic_at_k(kpoint, ibnd)
        

    @tools.timefn
    def get_bandstructure(
        self,
        prefix: str,
        kpaths_str: typing.List[str],
        quality: int = 0, 
        make_folder: bool = True
    ) -> bands.BandStructureResult:
        """return a BandStructureResult

        To plot band structure, call `bandresult.plot_data()`
        if `make_folder`, generate plotable script
        """
        unitcell = kpoints.UnitCell(self.cell)
        kpath = unitcell.get_kpath_from_path_string(kpaths_str, quality)
        band_result = bands.BandStructureResult.from_tightbinding_and_kpath(self.tb, kpath)

        e_min = np.min(band_result.E)
        e_max = np.max(band_result.E)
        e_min = e_min - (e_max - e_min) * 0.05
        e_max = e_max + (e_max - e_min) * 0.05

        ticks = band_result.ticks
        data_file = f"{prefix}_band.dat"
        plot_file = f"plot_{prefix}_band.py"
        result_file = f"{prefix}_band.pdf"

        if make_folder:
            folder_name = f"{prefix}_band"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            data_file_write = os.path.join(folder_name, data_file)
            plot_file_write = os.path.join(folder_name, plot_file)
        
            band_result.write_data_to_file(data_file_write)

            plotfile_lines = []
            plotfile_lines.append(f"import matplotlib.pyplot as plt")
            plotfile_lines.append(f"import numpy as np")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"data = np.loadtxt('{data_file}')")
            plotfile_lines.append(f"x = data[:,0]")
            plotfile_lines.append(f"ys = data[:, 1:]")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"ymin = {e_min:>.8f}")
            plotfile_lines.append(f"ymax = {e_max:>.8f}")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"fig = plt.figure()")
            plotfile_lines.append(f"axes = fig.subplots()")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"axes.set_ylabel('Energy (eV)')")
            plotfile_lines.append(f"axes.set_xlim(min(x), max(x))")
            plotfile_lines.append(f"axes.set_ylim(ymin, ymax)")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"for band in ys.T:")
            plotfile_lines.append(f"    axes.plot(x, band, '-', color = 'blue')")
            plotfile_lines.append(f"")
            plotfile_lines.append(
                f"tick_x = [" + ", ".join([f"{tic.xpos:.8f}" for tic in ticks]) + "]")
            plotfile_lines.append(
                f"tick_s = [" + ", ".join([f"'{tic.symbol}'" for tic in ticks])+ "]")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"assert np.isclose(tick_x[-1], max(x), atol=1e-4)")
            plotfile_lines.append(f"tick_x[-1] = max(x)")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"for tx in tick_x:")
            plotfile_lines.append(
                f"    axes.plot([tx, tx], [ymin, ymax], color = 'gray', alpha = 0.5)")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))")
            plotfile_lines.append(f"axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"fig.savefig('{result_file}')")

            with open(plot_file_write, 'w') as f:
                f.write("\n".join(plotfile_lines))
            
        return band_result


    @tools.timefn
    def get_dos(self,
        prefix: str, 
        emin: typing.Optional[float] = None,
        emax: typing.Optional[float] = None,
        gridsize: int = 30,
        xdensity: int = 100,
        make_folder: bool = True
    ) -> dos.TetraDOS:
        """Obtain the density of states

        if `make_folder`, we will generate the data file and a plotting 
        script in the folder called `prefix`.
        To plot data, call TetraDOS.plot
        """
        cell = kpoints.UnitCell(self.tb.cell)
        kmesh = dos.TetraKmesh(cell.reciprocal_cell, cell.recommend_kgrid(gridsize))

        energies = self.tb.solveE_at_ks(kmesh.kpoints)

        if emin is not None:
            e_min = emin
        else:
            e_min = np.min(energies)

        if emax is not None:
            e_max = emax
        else:
            e_max = np.max(energies)

        e_min = e_min - (e_max - e_min) * 0.05
        e_max = e_max + (e_max - e_min) * 0.05

        dos_result = dos.TetraDOS(kmesh, energies, np.linspace(e_min, e_max, xdensity))
        
        dos_file = f"{prefix}_dos.dat"
        plot_file = f"plot_{prefix}_dos.py"
        plot_result = f"{prefix}_dos.pdf"

        if make_folder:
            folder_name = f"{prefix}_dos"
            if not os.path.exists(folder_name): os.makedirs(folder_name)
            dos_file_write = os.path.join(folder_name, dos_file)
            plot_file_write = os.path.join(folder_name, plot_file)

            dos_result.write_data_to_file(dos_file_write)

            plotfile_lines = []
            plotfile_lines.append(f"import matplotlib.pyplot as plt")
            plotfile_lines.append(f"import numpy as np")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"data = np.loadtxt('{dos_file}')")
            plotfile_lines.append(f"x = data[:,0]")
            plotfile_lines.append(f"y = data[:,1]")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"fig = plt.figure()")
            plotfile_lines.append(f"axis = fig.subplots()")
            plotfile_lines.append(f"axis.set_xlabel('Energy (eV)')")
            plotfile_lines.append(f"axis.set_ylabel('Density of States (1/eV)')")
            ymin = 0.0
            ymax = np.max(dos_result.dos) * 1.2
            plotfile_lines.append(f"ymin = {ymin}; ymax = {ymax}")
            plotfile_lines.append(f"xmin = {e_min}, xmax = {e_max}")

            plotfile_lines.append(f"axis.set_ylim(ymin, ymax)")
            plotfile_lines.append(f"axis.set_xlim(xmin, xmax)")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"axis.plot(x, y)")
            plotfile_lines.append(f"")
            plotfile_lines.append(f"fig.savefig('{plot_result}')")
            plotfile_lines.append(f"")

            with open(plot_file_write, 'w') as f:
                f.write("\n".join(plotfile_lines))

        return dos_result