from ..plotting.interface import get_molecular_wavefunction_from_linear_combination
from ..plotting import DensityCubePlot
from ..interaction.SALCs import LinearCombination

__all__ = ["plot_molecular_wavefunction_from_linear_combination"]

def plot_molecular_wavefunction_from_linear_combination(
    lc: LinearCombination, 
    outputfilename: str,
    quality: str = "high"
) -> None:
    """
    a high level function to plot molecular wavefunction, quality is one of the following:
    - low
    - standard
    - high
    """
    mw = get_molecular_wavefunction_from_linear_combination(lc)
    cube_plot = DensityCubePlot([mw], quality)
    cube_plot.plot_to_file(outputfilename)