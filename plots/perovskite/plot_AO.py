from automaticTB.SALCs import LinearCombination, VectorSpace
from automaticTB.examples.AH3.structure import get_nncluster_AH3
from automaticTB.plotting import DensityCubePlot
import numpy as np

from automaticTB.plotting.adaptor import MolecularWavefunction, WavefunctionsOnSite, Wavefunction
import numpy as np
import typing

cell = np.eye(3) * 6.0
position_A = np.array([3,3,3])
bond_length = 1.4
w = np.sqrt(3) / 2
pos_H1 = position_A + np.array([1,0,0]) * bond_length
pos_H2 = position_A + np.array([-0.5,w,0]) * bond_length
pos_H3 = position_A + np.array([-0.5,w * -1.0,0]) * bond_length
zero = Wavefunction(l=0, m=0, coeff = 0.0)

results = {
    "H_a1'": MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [zero]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [Wavefunction(l=0, m=0, coeff = 1/np.sqrt(3))]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [Wavefunction(l=0, m=0, coeff = 1/np.sqrt(3))]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [Wavefunction(l=0, m=0, coeff = 1/np.sqrt(3))]), 
    ]),
    "H_E1" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [zero]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [Wavefunction(l=0, m=0, coeff = 2/np.sqrt(6))]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [Wavefunction(l=0, m=0, coeff = -1/np.sqrt(6))]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [Wavefunction(l=0, m=0, coeff = -1/np.sqrt(6))]), 
    ]),
    "H_E2" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [zero]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [Wavefunction(l=0, m=0, coeff = 0.0)]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [Wavefunction(l=0, m=0, coeff = 1/np.sqrt(2))]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [Wavefunction(l=0, m=0, coeff = -1/np.sqrt(2))]), 
    ]),
    "A_A1" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [Wavefunction(l=0, m=0, coeff = 1.0)]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [zero]), 
    ]),
    "A_A2" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [Wavefunction(l=1, m=0, coeff = 1.0)]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [zero]), 
    ]),
    "A_E1" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [Wavefunction(l=1, m=1, coeff = 1.0)]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [zero]), 
    ]),
    "A_E2" : MolecularWavefunction(cell, [
        WavefunctionsOnSite(position_A, 6, "C", [Wavefunction(l=1, m=-1, coeff = 1.0)]), 
        WavefunctionsOnSite(pos_H1, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H2, 1, "H", [zero]), 
        WavefunctionsOnSite(pos_H3, 1, "H", [zero]), 
    ]),
}

# l=0, m=0  -> s
# l=1, m=-1 -> y
# l=1, m= 0 -> z
# l=1, m= 1 -> x


for key, value in results.items():
    plot = DensityCubePlot([value], 'high')
    plot.plot_to_file(f"{key}.cube")
    
