from automaticTB.plotting.density import DensityCubePlot
from automaticTB.plotting.molecular_wavefunction import \
    MolecularWavefunction, Wavefunction, WavefunctionsOnSite
import numpy as np

"""
This script plots all the wavefunction of the first three shell. notice how isosurface of the 
wavefunction change its size depending on the n number. 

This script also specifies the relationship between m number and the orbital name
"""


wavefunctions_spd = {
    #              n  l   m  coefficients
    "1_s"     : (1, 0,  0, 1.0),
    "2_s"     : (2, 0,  0, 1.0),
    "2_py"    : (2, 1, -1, 1.0),
    "2_pz"    : (2, 1,  0, 1.0),
    "2_px"    : (2, 1,  1, 1.0),
    "3_s"     : (3, 0,  0, 1.0),
    "3_py"    : (3, 1, -1, 1.0),
    "3_pz"    : (3, 1,  0, 1.0),
    "3_px"    : (3, 1,  1, 1.0),
    "3_dxy"   : (3, 2, -2, 1.0),
    "3_dyz"   : (3, 2, -1, 1.0),
    "3_dz2"   : (3, 2,  0, 1.0),
    "3_dxz"   : (3, 2,  1, 1.0),
    "3_dx2-y2": (3, 2,  2, 1.0)
}

if __name__ == "__main__":
    """plot everything"""
    for key, value in wavefunctions_spd.items():
        wf = MolecularWavefunction(
            np.eye(3) * 6, 
            [
                WavefunctionsOnSite(
                    np.array([3.0, 3.0, 3.0]), 14, [Wavefunction(*value)]
                )
            ]
        )

        plot = DensityCubePlot([wf])
        plot.plot_to_file(f"{key}.cube")
