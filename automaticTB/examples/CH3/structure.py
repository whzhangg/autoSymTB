import numpy as np
from automaticTB.structure import CrystalSite, NearestNeighborCluster
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from DFTtools.SiteSymmetry.site_symmetry_group import get_point_group_as_SiteSymmetryGroup

group = get_point_group_as_SiteSymmetryGroup("3m")
w = np.sqrt(3) / 2
crystalsites = [
    CrystalSite.from_data(6, np.array([0.0, 0.0, 0.0]), 0, np.array([0,0,0])),
    CrystalSite.from_data(1, np.array([0.0, 1.0, 0.0]), 1, np.array([0,0,0])),
    CrystalSite.from_data(1, np.array([  w,-0.5, 0.0]), 2, np.array([0,0,0])),
    CrystalSite.from_data(1, np.array([-1 * w,-0.5, 0.0]), 3, np.array([0,0,0])),
]
orbitals = OrbitalsList([Orbitals([0,1]), Orbitals([0]), Orbitals([0]), Orbitals([0])])

nncluster = NearestNeighborCluster(crystalsites, orbitals, group)
