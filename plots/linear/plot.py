from automaticTB.plotting.adaptor import MolecularWavefunction, WavefunctionsOnSite, Wavefunction
from automaticTB.plotting import DensityCubePlot
import numpy as np
import typing

cell = np.eye(3) * 10.0

def get_wavefunctions() -> typing.Dict[str, MolecularWavefunction]:
    s = Wavefunction(l=0, m=0, coeff = 1.0)
    ms = Wavefunction(l=0, m=0, coeff = -1.0)
    pz = Wavefunction(l=1, m=1, coeff = 1.0)
    zero = Wavefunction(l=0, m=0, coeff = 0.0)
    result = {}
    result["s_A'"] = MolecularWavefunction(
        cell, 
        [WavefunctionsOnSite(np.array([2, 5, 5]), 6, "C", [s]),
         WavefunctionsOnSite(np.array([8, 5, 5]), 6, "C", [s]),
         WavefunctionsOnSite(np.array([5, 5, 5]), 6, "C", [zero])
        ]
    )
    result["s_A''"] = MolecularWavefunction(
        cell, 
        [WavefunctionsOnSite(np.array([2, 5, 5]), 6, "C", [ms]),
         WavefunctionsOnSite(np.array([8, 5, 5]), 6, "C", [s]),
         WavefunctionsOnSite(np.array([5, 5, 5]), 6, "C", [zero])
        ]
    )
    result["pz"] = MolecularWavefunction(
        cell, 
        [WavefunctionsOnSite(np.array([2, 5, 5]), 6, "C", [zero]),
         WavefunctionsOnSite(np.array([8, 5, 5]), 6, "C", [zero]),
         WavefunctionsOnSite(np.array([5, 5, 5]), 6, "C", [pz])
        ]
    )
    return result

for key, value in get_wavefunctions().items():
    plot = DensityCubePlot([value], 'high')
    plot.plot_to_file(f"{key}.cube")
    
    