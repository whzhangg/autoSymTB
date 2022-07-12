from automaticTB.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from automaticTB.SALCs import VectorSpace
from automaticTB.SALCs.stable_decompose import decompose_vectorspace_to_namedLC

def get_Si_cluster():
    Si_structure_file = "../../examples/Si/Si_ICSD51688.cif"
    c, p, t = read_generic_to_cpt(Si_structure_file)
    Si_structure = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.0)
    nn_cluster = Si_structure.nnclusters[0]
    return nn_cluster

def test_decomposition():
    nncluster = get_Si_cluster()
    vectorspace = VectorSpace.from_NNCluster(nncluster)
    namedlc = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)

