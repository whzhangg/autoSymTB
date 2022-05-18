from automaticTB.examples.Perovskite.structure import cell, types, positions, orbits_used

from automaticTB.structure.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from DFTtools.config import rootfolder
import os
from automaticTB.SALCs.vectorspace import get_vectorspace_from_NNCluster
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.hamiltionian.interaction import RandomInteraction, Interaction
from automaticTB.hamiltionian.MOcoefficient import MOCoefficient, get_AOlists_from_crystalsites_orbitals
from automaticTB.hamiltionian.tightbinding_data import make_HijR_list
from automaticTB.hamiltionian.model import TightBindingModel
from automaticTB.utilities import print_matrix
from model_interaction import obtain_AO_interaction_from_AOlists
import numpy as np

structure: Structure = Structure.from_cpt_rcut(cell, positions, types, 3.0)  

c, p, t = structure.cpt
# distance Pb - Cl is ~2.8 A

interactions = []

for cluster in structure.nnclusters:
    assert len(cluster.crystalsites) in [3, 7]
    group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
    assert group.groupname in ["m-3m", "4/mmm"]
    
    vectorspace = get_vectorspace_from_NNCluster(cluster, orbits_used)
    orbitals = vectorspace.orbitals
    crystalsiteswithsymmetry = cluster.get_CrystalSites_and_Equivalence_from_group(group)

    aos = get_AOlists_from_crystalsites_orbitals(cluster.crystalsites, orbitals)
    interaction = obtain_AO_interaction_from_AOlists(c, p, aos)
    interactions.append(interaction)
    assert np.allclose(interaction.interactions, interaction.interactions.T)
    #if len(cluster.crystalsites) == 7: continue

    #print(cluster.crystalsites)
    #decomposed_vectorspace = decompose_vectorspace(vectorspace, group)
    #mos = MOCoefficient(crystalsiteswithsymmetry, decomposed_vectorspace)
    #aolist = mos.AOs


hijr_list = make_HijR_list(interactions)


model = TightBindingModel(c, p, t, hijr_list)
#print_matrix(model.Hijk((0.0,0.0,0.0)).real, "{:>6.2f}")
print(model.solveE_at_k((0.0,0,0)))
print(model.solveE_at_k((0.5,0,0)))
