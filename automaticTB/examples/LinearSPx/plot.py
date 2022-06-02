from automaticTB.plotting.adaptor import MolecularWavefunction, WavefunctionsOnSite, Wavefunction
from automaticTB.plotting import DensityCubePlot
import numpy as np


def get_wavefunctions():
    s = Wavefunction(l=0, m=0, coeff = 1.0)
    px = Wavefunction(l=1, m=1, coeff = 1.0)
    mpx = Wavefunction(l=1, m=1, coeff = -1.0)

    sites = [
        WavefunctionsOnSite(np.array([2, 5, 2.5]), 6, "C", [px]),
        WavefunctionsOnSite(np.array([5, 5, 2.5]), 6, "C", [s]),
        WavefunctionsOnSite(np.array([8, 5, 2.5]), 6, "C", [px]),
        WavefunctionsOnSite(np.array([2, 5, 6.5]), 6, "C", [mpx]),
        WavefunctionsOnSite(np.array([5, 5, 6.5]), 6, "C", [s]),
        WavefunctionsOnSite(np.array([8, 5, 6.5]), 6, "C", [px]),
    ]

    return MolecularWavefunction(
        np.eye(3) * 10.0,
        sites
    )

plot = DensityCubePlot([get_wavefunctions()], 'high')
plot.plot_to_file("s-px.cube")