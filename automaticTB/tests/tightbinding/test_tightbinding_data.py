from automaticTB.structure.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from DFTtools.config import rootfolder
import os
from automaticTB.SALCs.vectorspace import get_vectorspace_from_NNCluster
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.tightbinding.MOcoefficient import MOCoefficient, InteractionMatrix, AO_from_MO
#from automaticTB.hamiltionian.NNInteraction import get_interaction_matrix_from_MO_and_interaction
#from automaticTB.tightbinding.model_generation import generate_tightbinding_data
from automaticTB.tightbinding.model import TightBindingModel
from automaticTB.tightbinding.tightbinding_data import make_HijR_list


Si_cif_fn = os.path.join(rootfolder, "tests/structures/Si_ICSD51688.cif")

def test_load_structure_cutoff():
    c,p,t = read_generic_to_cpt(Si_cif_fn)
    structure: Structure = Structure.from_cpt_rcut(c, p, t, 3.0)

    interactions = []

    for cluster in structure.nnclusters:
        vectorspace = get_vectorspace_from_NNCluster(cluster, {"Si": "s"})
        group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
        decomposed_vectorspace = decompose_vectorspace(vectorspace, group)

        crystalsiteswithsymmetry = cluster.get_CrystalSites_and_Equivalence_from_group(group)
        mos = MOCoefficient(crystalsiteswithsymmetry, decomposed_vectorspace)
        MOinteraction = InteractionMatrix.random_from_states(mos.MOs)
        AOinteraction = AO_from_MO(mos, MOinteraction)
        interactions.append(AOinteraction)

    HijR_list = make_HijR_list(interactions)
    c, p, t = structure.cpt

    model = TightBindingModel(c,p,t,HijR_list)
    print(model.Hijk((0.5,0,0)))
    print(model.solveE_at_k((0.5,0,0)))


test_load_structure_cutoff()