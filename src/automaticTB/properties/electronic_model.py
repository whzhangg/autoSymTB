import typing, numpy as np
from .tightbinding import TightBindingModel
from .kpoints import UnitCell
from .bands import BandStructureResult
from .functions.core_properties import (
    write_DOS_from_tightbinding, wrtie_bandstructure_from_tightbinding
)

class ElectronicModel:
    """
    this class is a main wrapper to extract properties from a tightbinding model
    """
    def __init__(self, tbmodel: TightBindingModel) -> None:
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


    def plot_bandstructure(
        self, 
        prefix: str,
        kpaths_str: typing.List[str]
    ) -> None:
        """
        plot band structure result directly
        """
        unitcell = UnitCell(self.cell)
        kpath = unitcell.get_kpath_from_path_string(kpaths_str)
        bandresult = BandStructureResult.from_tightbinding_and_kpath(self.tb, kpath)
        filename = f"{prefix}.pdf"
        bandresult.plot_data(filename)


    def write_bandstructure(
        self,
        prefix: str,
        kpaths_str: typing.List[str],
        emin: typing.Optional[float] = None,
        emax: typing.Optional[float] = None,
        quality: int = 0, 
        make_folder: bool = True
    ) -> None:
        """
        call functions.core_properties.wrtie_bandstructure_from_tightbinding
        """
        wrtie_bandstructure_from_tightbinding(
            self.tb, 
            prefix, kpaths_str, emin, emax, quality, make_folder
        )


    def write_dos(self,
        prefix: str, 
        emin: typing.Optional[float] = None,
        emax: typing.Optional[float] = None,
        gridsize: int = 30,
        xdensity: int = 100,
        make_folder: bool = True
    ) -> None:
        """
        call functions.core_properties.write_DOS_from_tightbinding
        """
        write_DOS_from_tightbinding(
            self.tb,
            prefix, emin, emax, gridsize, xdensity, make_folder
        )