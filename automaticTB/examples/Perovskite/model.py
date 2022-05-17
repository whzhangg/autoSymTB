from automaticTB.examples.Perovskite.structure import cell, types, positions, orbits_used

from automaticTB.structure.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from DFTtools.config import rootfolder
import os
from automaticTB.SALCs.vectorspace import get_vectorspace_from_NNCluster
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.hamiltionian.interaction import RandomInteraction, Interaction
from automaticTB.hamiltionian.MOcoefficient import MOCoefficient, InteractingLCPair
from automaticTB.hamiltionian.NNInteraction import get_interaction_matrix_from_MO_and_interaction
from automaticTB.tightbinding.model_generation import generate_tightbinding_data
from automaticTB.tightbinding.model import TightBindingModel
from automaticTB.utilities import print_matrix

def non_interacting(pair: InteractingLCPair):
    name1 = pair.rep1
    name2 = pair.rep2
    if name1.irreps != name2.irreps:
        return True


structure: Structure = Structure.from_cpt_rcut(cell, positions, types, 3.0)  
# distance Pb - Cl is ~2.8 A

interactions = []

for cluster in structure.nnclusters:
    assert len(cluster.crystalsites) in [3, 7]
    group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
    assert group.groupname in ["m-3m", "4/mmm"]
    
    vectorspace = get_vectorspace_from_NNCluster(cluster, orbits_used)
    crystalsiteswithsymmetry = cluster.get_CrystalSitesWithSymmetry_from_group(group)

    if len(cluster.crystalsites) == 7: continue
    print(cluster.crystalsites)
    decomposed_vectorspace = decompose_vectorspace(vectorspace, group)
    mos = MOCoefficient(crystalsiteswithsymmetry, decomposed_vectorspace)
    print(mos.all_aos)
    #for i in range(mos.num_MOpairs):
    #    pair = mos.get_paired_lc_with_name_by_index(i)
    #    if not non_interacting(pair):
    #        print(pair.rep1)
    #        print(pair.lc1)
    #        print(pair.rep2)
    #        print(pair.lc2)
    #        print()
    #raise
    orb_and_interaction = get_interaction_matrix_from_MO_and_interaction(
        mos, RandomInteraction()
    )
    print_matrix(orb_and_interaction.interaction, "{:>5.2f}")
    
    #interactions.append(orb_and_interaction)
    raise

raise

c, p, t = structure.cpt
tbd = generate_tightbinding_data(c, p, t, structure.nnclusters, interactions)
for hij in tbd.HijR_list:
    print(hij)

model = TightBindingModel(tbd)
print(model.Hijk((0.5,0,0)))
print(model.solveE_at_k((0.5,0,0)))
