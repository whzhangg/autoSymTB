import numpy as np
from ...structure import LocalSite, CrystalSite, NeighborCluster
from ...sitesymmetry import get_point_group_as_SiteSymmetryGroup

__all__ = ["get_AH3_nncluster"]

def get_AH3_nncluster() -> NeighborCluster:
    """
    return a molecular cluster centered on A atom
    """
    group = get_point_group_as_SiteSymmetryGroup("-6m2")
    w = np.sqrt(3) / 2
    center_crystalsite = CrystalSite(
        LocalSite(6, np.array([0.0, 0.0, 0.0]) * 2.0),
        np.array([0.0, 0.0, 0.0]) * 2.0,
        0, np.zeros(3)
    )
    other_crystalsites = [
        CrystalSite(
            LocalSite(1, np.array([0.0, 1.0, 0.0]) * 2.0),
            np.array([0.0, 1.0, 0.0]) * 2.0,
            1, np.zeros(3)
        ),
        CrystalSite(
            LocalSite(1, np.array([  w,-0.5, 0.0]) * 2.0),
            np.array([  w,-0.5, 0.0]) * 2.0,
            2, np.zeros(3)
        ),
        CrystalSite(
            LocalSite(1, np.array([-1 * w,-0.5, 0.0]) * 2.0),
            np.array([-1 * w,-0.5, 0.0]) * 2.0,
            3, np.zeros(3)
        ),
    ]

    return NeighborCluster(center_crystalsite, other_crystalsites, group)
