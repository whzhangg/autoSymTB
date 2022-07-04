import os
from automaticTB.structure import Structure
from automaticTB.tools import read_cif_to_cpt
import numpy as np

def get_si_cpt_orbref():
    cwd,_ = os.path.split(__file__)
    structure_ci_cif = os.path.join(cwd, "../examples/Si/Si_ICSD51688.cif")
    c, p, t = read_cif_to_cpt(structure_ci_cif)
    orb_ref = {"Si": [0,1]}

    return c, p, t, orb_ref

def get_perovskite_cpt_orbref():
    cell = np.eye(3, dtype=float) * 5.6804
    types = [ 82, 17, 17, 17 ]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5]
    ])
    orbits_used = {
        "Pb": [0, 1],
        "Cl": [0, 1]
    }
    return cell, positions, types, orbits_used

def test_cutoff_include_in_voronoi_si():
    c, p, t, orb_ref = get_si_cpt_orbref()
    structure_cutoff = Structure.from_cpt_rcut(c,p,t,orb_ref, 3.0)
    structure_voronoi = Structure.from_cpt_voronoi(c,p,t,orb_ref, 1e-1, 3.0)

    assert len(structure_cutoff.nnclusters) == len(structure_voronoi.nnclusters)

    for cluster1, cluster2 in zip(structure_cutoff.nnclusters, structure_voronoi.nnclusters):
        for site1 in cluster1.crystalsites:
            assert site1 in cluster2.crystalsites

def test_cutoff_include_in_voronoi_perovskite():
    c, p, t, orb_ref = get_perovskite_cpt_orbref()
    structure_cutoff = Structure.from_cpt_rcut(c,p,t,orb_ref, 3.0)
    structure_voronoi = Structure.from_cpt_voronoi(c,p,t,orb_ref, 1e-1, 3.0)

    assert len(structure_cutoff.nnclusters) == len(structure_voronoi.nnclusters)

    for cluster1, cluster2 in zip(structure_cutoff.nnclusters, structure_voronoi.nnclusters):
        for site1 in cluster1.crystalsites:
            assert site1 in cluster2.crystalsites
