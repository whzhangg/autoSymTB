import numpy as np
import dataclasses, typing
from automaticTB.structure import CrystalSite, NearestNeighborCluster, Site
from automaticTB.atomic_orbitals import Orbitals, OrbitalsList
from automaticTB.sitesymmetry import get_point_group_as_SiteSymmetryGroup, SiteSymmetryGroup
from automaticTB.SALCs import VectorSpace, decompose_vectorspace_to_namedLC
from automaticTB.interaction import InteractionEquation
from automaticTB.printing import print_ao_pairs
from automaticTB.plotting import DensityCubePlot

sp = Orbitals([0,1])
Ri = 1.4
A = 14
B = 8
C = 9
center = np.zeros(3)
translation = np.zeros(3)
w = np.sqrt(3) / 2

def get_symbol():
    for key, value in coordinations.items():
        print(key)
        group = get_point_group_as_SiteSymmetryGroup(value[1])
        print(group.dressed_op.keys())


def symmetry_is_correct(baresites: typing.List[Site], group: SiteSymmetryGroup) -> bool:
    for op in group.operations:
        new_sites_are_in = [site.rotate(op) in baresites for site in baresites]
        if not all(new_sites_are_in):
            return False
    return True


def get_NNC_from_SimpleSites(sites: typing.List[Site], groupname: str) -> NearestNeighborCluster:
    group = get_point_group_as_SiteSymmetryGroup(groupname)
    assert symmetry_is_correct(sites, group)
    print("symmetry correct")
    crystalsites = [ 
        CrystalSite.from_data(s.atomic_number, s.pos, i, translation) for i,s in enumerate(sites)
    ]
    orbitals = OrbitalsList([sp] * len(crystalsites))
    return NearestNeighborCluster(crystalsites, orbitals, group)


coordinations = {
    "square bipyramidal": (
        [
            Site(A, Ri * center), 
            Site(B, Ri * np.array([0.0, 0.0, 1.7])),
            Site(B, Ri * np.array([0.0, 0.0,-1.7])),
            Site(C, Ri * np.array([1.0, 1.0, 0.0])),
            Site(C, Ri * np.array([-1.0, 1.0, 0.0])),
            Site(C, Ri * np.array([1.0, -1.0, 0.0])),
            Site(C, Ri * np.array([-1.0, -1.0, 0.0])),
        ], "4/mmm"
    ),
}

lc_to_print = [
    "B_1u @ 1^th B_1u",
    "A_1g @ 3^th A_1g",
    "E_u->B_1 @ 3^th E_u"
]

def get_cifs():
    for key, value in coordinations.items():
        fn = f"{key}.cif"
        s, g = value
        nncluster = get_NNC_from_SimpleSites(s, g)
        nncluster.write_to_cif(fn)


def print_free_parameters():
    for key, value in coordinations.items():
        print(key)
        s, g = value
        nncluster = get_NNC_from_SimpleSites(s, g)
        vectorspace = VectorSpace.from_NNCluster(nncluster)

        named_lcs = decompose_vectorspace_to_namedLC(vectorspace, nncluster.sitesymmetrygroup)
        for nlc in named_lcs:
            print(nlc.name)
            print(nlc.lc)
            if str(nlc.name) in lc_to_print:
                plot = DensityCubePlot.from_linearcombinations([nlc.lc], "high")
                plot.plot_to_file(f"{str(nlc.name)}.cube")
        #free_pairs = InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs).free_AOpairs
        #print_ao_pairs(nncluster, free_pairs)
        #print(len(free_pairs))
        break

def print_total_number_parameters():
    for key, value in coordinations.items():
        s, g = value
        print(f"{key}, , {4 * 4 * (len(s)-1)}")

if __name__ == "__main__":
    get_cifs()
    print_free_parameters()
    print_total_number_parameters()

"""
    Output
linear
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  4 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  8 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C pz -> C  s @ (  0.00,  0.00, -1.00)
  3 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00, -1.00)
  5 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00, -1.00)
  6 > Free AO interaction: C  s -> C pz @ (  0.00,  0.00, -1.00)
  7 > Free AO interaction: C px -> C px @ (  0.00,  0.00, -1.00)
8
trigonal plane
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
  6 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  3 > Free AO interaction: C  s -> C  s @ ( -0.87, -0.50,  0.00)
  4 > Free AO interaction: C pz -> C pz @ ( -0.87, -0.50,  0.00)
  5 > Free AO interaction: C  s -> C px @ ( -0.87, -0.50,  0.00)
  7 > Free AO interaction: C px -> C  s @ ( -0.87, -0.50,  0.00)
  8 > Free AO interaction: C px -> C py @ ( -0.87, -0.50,  0.00)
  9 > Free AO interaction: C px -> C px @ ( -0.87, -0.50,  0.00)
9
tetrahedral
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  7 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C px -> C  s @ ( -1.00, -1.00,  1.00)
  3 > Free AO interaction: C px -> C pz @ ( -1.00, -1.00,  1.00)
  4 > Free AO interaction: C px -> C px @ ( -1.00, -1.00,  1.00)
  5 > Free AO interaction: C  s -> C  s @ ( -1.00, -1.00,  1.00)
  6 > Free AO interaction: C  s -> C px @ ( -1.00, -1.00,  1.00)
7
square plane
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
  9 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  3 > Free AO interaction: C px -> C  s @ ( -1.00, -1.00,  0.00)
  4 > Free AO interaction: C px -> C py @ ( -1.00, -1.00,  0.00)
  5 > Free AO interaction: C px -> C px @ ( -1.00, -1.00,  0.00)
  6 > Free AO interaction: C  s -> C  s @ ( -1.00, -1.00,  0.00)
  7 > Free AO interaction: C  s -> C px @ ( -1.00, -1.00,  0.00)
  8 > Free AO interaction: C pz -> C pz @ ( -1.00, -1.00,  0.00)
9
square rectangle
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  5 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
 12 > Free AO interaction: C py -> C py @ (  0.00,  0.00,  0.00)
 14 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C py -> C  s @ ( -1.00, -0.80,  0.00)
  3 > Free AO interaction: C py -> C py @ ( -1.00, -0.80,  0.00)
  4 > Free AO interaction: C py -> C px @ ( -1.00, -0.80,  0.00)
  6 > Free AO interaction: C px -> C  s @ ( -1.00, -0.80,  0.00)
  7 > Free AO interaction: C px -> C py @ ( -1.00, -0.80,  0.00)
  8 > Free AO interaction: C px -> C px @ ( -1.00, -0.80,  0.00)
  9 > Free AO interaction: C  s -> C  s @ ( -1.00, -0.80,  0.00)
 10 > Free AO interaction: C  s -> C py @ ( -1.00, -0.80,  0.00)
 11 > Free AO interaction: C  s -> C px @ ( -1.00, -0.80,  0.00)
 13 > Free AO interaction: C pz -> C pz @ ( -1.00, -0.80,  0.00)
14
square pyramidal
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C  s -> C pz @ (  0.00,  0.00,  0.00)
  8 > Free AO interaction: C pz -> C  s @ (  0.00,  0.00,  0.00)
  9 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
 15 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  3 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  1.00)
  4 > Free AO interaction: C  s -> C pz @ (  0.00,  0.00,  1.00)
  5 > Free AO interaction: C  s -> C  s @ ( -1.00, -1.00,  0.00)
  6 > Free AO interaction: C  s -> C pz @ ( -1.00, -1.00,  0.00)
  7 > Free AO interaction: C  s -> C px @ ( -1.00, -1.00,  0.00)
 10 > Free AO interaction: C pz -> C  s @ (  0.00,  0.00,  1.00)
 11 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  1.00)
 12 > Free AO interaction: C pz -> C  s @ ( -1.00, -1.00,  0.00)
 13 > Free AO interaction: C pz -> C pz @ ( -1.00, -1.00,  0.00)
 14 > Free AO interaction: C pz -> C px @ ( -1.00, -1.00,  0.00)
 16 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  1.00)
 17 > Free AO interaction: C px -> C  s @ ( -1.00, -1.00,  0.00)
 18 > Free AO interaction: C px -> C py @ ( -1.00, -1.00,  0.00)
 19 > Free AO interaction: C px -> C pz @ ( -1.00, -1.00,  0.00)
 20 > Free AO interaction: C px -> C px @ ( -1.00, -1.00,  0.00)
20
cubic
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  6 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C  s -> C  s @ ( -1.00, -1.00, -1.00)
  3 > Free AO interaction: C  s -> C px @ ( -1.00, -1.00, -1.00)
  4 > Free AO interaction: C px -> C  s @ ( -1.00, -1.00, -1.00)
  5 > Free AO interaction: C px -> C pz @ ( -1.00, -1.00, -1.00)
  7 > Free AO interaction: C px -> C px @ ( -1.00, -1.00, -1.00)
7
trigonal prism
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C pz -> C pz @ (  0.00,  0.00,  0.00)
 10 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  3 > Free AO interaction: C px -> C  s @ ( -0.87, -0.50, -0.90)
  4 > Free AO interaction: C px -> C py @ ( -0.87, -0.50, -0.90)
  5 > Free AO interaction: C px -> C pz @ ( -0.87, -0.50, -0.90)
  6 > Free AO interaction: C px -> C px @ ( -0.87, -0.50, -0.90)
  7 > Free AO interaction: C pz -> C  s @ ( -0.87, -0.50, -0.90)
  8 > Free AO interaction: C pz -> C pz @ ( -0.87, -0.50, -0.90)
  9 > Free AO interaction: C pz -> C px @ ( -0.87, -0.50, -0.90)
 11 > Free AO interaction: C  s -> C  s @ ( -0.87, -0.50, -0.90)
 12 > Free AO interaction: C  s -> C pz @ ( -0.87, -0.50, -0.90)
 13 > Free AO interaction: C  s -> C px @ ( -0.87, -0.50, -0.90)
13
octahedral
symmetry correct
Free interaction parameters: 
  1 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00,  0.00)
  3 > Free AO interaction: C px -> C px @ (  0.00,  0.00,  0.00)
  2 > Free AO interaction: C px -> C px @ (  0.00,  0.00, -1.00)
  4 > Free AO interaction: C  s -> C  s @ (  0.00,  0.00, -1.00)
  5 > Free AO interaction: C  s -> C pz @ (  0.00,  0.00, -1.00)
  6 > Free AO interaction: C px -> C  s @ ( -1.00,  0.00,  0.00)
  7 > Free AO interaction: C px -> C px @ ( -1.00,  0.00,  0.00)
7
"""