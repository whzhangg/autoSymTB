from automaticTB.structure.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from DFTtools.config import rootfolder
import os
from automaticTB.SALCs.vectorspace import get_vectorspace_from_NNCluster
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.hamiltionian.interaction import RandomInteraction
from automaticTB.hamiltionian.MOcoefficient import MOCoefficient
from automaticTB.hamiltionian.NNInteraction import get_interaction_matrix_from_MO_and_interaction
from automaticTB.tightbinding.model_generation import generate_tightbinding_data
from automaticTB.tightbinding.model import TightBindingModel


Si_cif_fn = os.path.join(rootfolder, "tests/structures/Si_ICSD51688.cif")

def test_load_structure_cutoff():
    c,p,t = read_generic_to_cpt(Si_cif_fn)
    structure: Structure = Structure.from_cpt_rcut(c, p, t, 3.0)

    interactions = []

    for cluster in structure.nnclusters:
        vectorspace = get_vectorspace_from_NNCluster(cluster, {"Si": "s"})
        group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
        decomposed_vectorspace = decompose_vectorspace(vectorspace, group)

        crystalsiteswithsymmetry = cluster.get_CrystalSitesWithSymmetry_from_group(group)
        mos = MOCoefficient(crystalsiteswithsymmetry, decomposed_vectorspace)
        orb_and_interaction = get_interaction_matrix_from_MO_and_interaction(
            mos, RandomInteraction()
        )
        print(orb_and_interaction.ilms)
        interactions.append(orb_and_interaction)

    c, p, t = structure.cpt
    tbd = generate_tightbinding_data(c, p, t, structure.nnclusters, interactions)
    for hij in tbd.HijR_list:
        print(hij)

    model = TightBindingModel(tbd)
    print(model.Hijk((0.5,0,0)))
    print(model.solveE_at_k((0.5,0,0)))


test_load_structure_cutoff()