import numpy as np
from ...structure import CrystalSite, NearestNeighborCluster
from ...atomic_orbitals import Orbitals, OrbitalsList
from ...sitesymmetry import get_point_group_as_SiteSymmetryGroup

def get_nncluster_AH3():
    group = get_point_group_as_SiteSymmetryGroup("-6m2")
    w = np.sqrt(3) / 2
    crystalsites = [
        CrystalSite.from_data(6, np.array([0.0, 0.0, 0.0]) * 2.0, 0, np.array([0,0,0])),
        CrystalSite.from_data(1, np.array([0.0, 1.0, 0.0]) * 2.0, 1, np.array([0,0,0])),
        CrystalSite.from_data(1, np.array([  w,-0.5, 0.0]) * 2.0, 2, np.array([0,0,0])),
        CrystalSite.from_data(1, np.array([-1 * w,-0.5, 0.0]) * 2.0, 3, np.array([0,0,0])),
    ]
    orbitals = OrbitalsList([Orbitals([0,1]), Orbitals([0]), Orbitals([0]), Orbitals([0])])

    return NearestNeighborCluster(crystalsites, orbitals, group)
