import numpy as np
import os, typing
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.structure import NearestNeighborCluster, Structure
from automaticTB.interaction import get_free_interaction_AO, InteractionEquation
from automaticTB.atomic_orbitals import AO
from automaticTB.examples.Perovskite.structure import get_perovskite_structure
from automaticTB.printing import print_ao_pairs, print_matrix, print_InteractionPairs
from automaticTB.examples.Perovskite.ao_interaction import get_interaction_values_from_list_AOpairs
from automaticTB.plotting import DensityCubePlot
from automaticTB.plotting.adaptor import MolecularWavefunction, Wavefunction, WavefunctionsOnSite
from automaticTB.tools import write_yaml, read_yaml


zero = Wavefunction(l=0, m=0, coeff = 0.0)
s = Wavefunction(l=0, m=0, coeff = 1.0)
px = Wavefunction(l=1, m=-1, coeff = 1.0)
pz = Wavefunction(l=1, m=1, coeff = 1.0)
cell = np.eye(3) * 11.3608

plot = {
    "pb s": MolecularWavefunction(
        cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [s]), 
        ]
    ),
    "pb px": MolecularWavefunction(
        cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [px]), 
        ]
    ),
    "pb pz": MolecularWavefunction(
        cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [pz]), 
        ]
    ),
    "Cl s": MolecularWavefunction(
        cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [s]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [zero]), 
        ]
    ),
    "Cl px": MolecularWavefunction(
        mw.cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [px]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [zero]), 
        ]
    ),
    "Cl pz": MolecularWavefunction(
        cell, [ 
            WavefunctionsOnSite(np.array([2.8402, 5.6804, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 2.8402, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 2.8402]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 8.5206]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([5.6804, 8.5206, 5.6804]), 17, "Cl", [zero]), 
            WavefunctionsOnSite(np.array([8.5206, 5.6804, 5.6804]), 17, "Cl", [pz]), 
            WavefunctionsOnSite(np.array([5.6804, 5.6804, 5.6804]), 82, "Pb", [zero]), 
        ]
    )
}


for key,val in plot.items():
    plot = DensityCubePlot([val], "high")
    plot.plot_to_file(f"{key}.cube")