from automaticTB.examples.structures import fcc_si
from automaticTB.structure import Structure

def test_cluster_cutoff():
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.9)
    for cluster in struct.nnclusters:
        assert cluster.num_sites == 17


def test_cluster_voronoi():
    c, p, t = fcc_si["cell"], fcc_si["positions"], fcc_si["types"]
    struct = Structure.from_cpt_voronoi(c,p,t, {"Si": [0,1]}, rcut=4.0)
    for cluster in struct.nnclusters:
        assert cluster.num_sites == 5
