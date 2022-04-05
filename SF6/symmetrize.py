import spglib
from DFTtools.DFT.output_parser import SCFout
import numpy as np
from copy import deepcopy
from e3nn.o3 import Irreps, Irrep
from numpy import pi
import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Symmetry:
    euler_angle : Tuple[float]
    inversion : int


def test_o3_basisfunction(lmax = 2):
    """
    In e3nn, we can verify that the basis function for l = 0, 1, 2 are in the following order, 
    note that z and y axis are exchanged.
    s x y z xz xy y2 yz x2-z2
    we can let it to be 
    s x z y xy xz z2 zy x2-y2
    and by dummy interchanging the y and z axis. so that the rotation is 
    gamma-z axis, beta_x axis and alpha-z axis,

    the x y z axis of the rotation is not the body axis, they do not rotate with the body
    """
    assumption = "s x y z xz xy y2 yz x2-y2"
    test_rotation = [
        # alpha, beta, gamma
        #     y,    x,     y axis
        # gamma is applied the first, alpha the last: R(abc) = R(a) * R(b) * R(c)
        # 
        (torch.tensor(pi/2, dtype = float), torch.tensor( 0.0, dtype = float), torch.tensor( 0.0, dtype = float)),
        (torch.tensor( 0.0, dtype = float), torch.tensor(pi/2, dtype = float), torch.tensor( 0.0, dtype = float)),
        (torch.tensor(-pi/2, dtype = float), torch.tensor(-pi/2, dtype = float), torch.tensor(pi/2, dtype = float)),
    ]
    irrep = Irreps.spherical_harmonics(lmax = lmax)
    for rot in test_rotation:
        print_format = "{:>7.3f}"*irrep.dim
        matrix = irrep.D_from_angles(*rot).numpy()
        for row in matrix:
            print(print_format.format(*row))
        print()

def find_symmetry(struct: tuple) -> list:
    # return rotation in cartesian coordinates
    dataset = spglib.get_symmetry_dataset(struct)
    rotations_crystal = dataset["rotations"]
    cellT = struct[0].T
    rotations_cartesian = [ cellT.dot(r).dot(np.linalg.inv(cellT)) for r in rotations_crystal ]
    return rotations_cartesian

rotations_D3h = [
    Symmetry((0,0,0), 0), # identity
    Symmetry((0,0,pi), 1), # horizontal refection
    Symmetry((0,0,2*pi/3), 0), # 120degree rotation
    Symmetry((0,0,  pi/3), 1), # S3
    Symmetry((0,pi,0), 0), # C2: rotation around x 
    Symmetry((0,pi,-pi), 1) # vertical ?
]

D3h_positions = [
    [ 0.0,   0.0, 0.0], 
    [ 2.0,   0.0, 0.0],
    [-1.0, np.sqrt(3), 0.0],
    [-1.0,-np.sqrt(3), 0.0],
]

# AO representation
class AORepresentation:
    """
    passing a tuple and a maximum l, we construct 
    the representation in the atomic orbital vector space
    with dimension natom * nbasis, 
    we define the sequence of the atomic orbital following e3nn
    s x z y xy xz z2 zy x2-y2
    and rotations (alpha, beta, gamma) to be around z, x, z axis 
    """
    defined_orbitals = "s x z y xy xz z2 zy x2-y2".split()
    def __init__(self, positions, lmax: int, symmetries: List[Symmetry]) -> None:
        self.positions = torch.tensor(positions)
        self.symmetry_operations = symmetries
        self.irrep = Irreps.spherical_harmonics(lmax = lmax)
        self.check_symmetry()

    def generate_representation(self):
        for sym in self.symmetry_operations:
            break

    def check_symmetry(self):
        # check if symmetry is indeed the same
        #cartesian_positions = np.einsum("ij, kj -> ki", self.cpt[0],self.cpt[1])
        cartesian_positions = self.positions
        pos_list = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in cartesian_positions.detach()])
        for symmetry in self.symmetry_operations:
            transformed = self.apply_symmetry(1, cartesian_positions, symmetry)
            transformed = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in transformed.detach()])
            assert transformed == pos_list, (transformed, symmetry)

    def apply_symmetry(self, l:int, values: torch.tensor, sym:Symmetry) -> torch.tensor:
        irrep = Irrep(l,(-1)**l)
        if l == 0:
            permutation = [0]
            reverse = [0]
        elif l == 1:
            permutation = [0, 2, 1] 
            reverse = [0, 2, 1]
        elif l == 2:
            permutation = [0,1,2,3,4]
            reverse= [0,1,2,3,4]
        rotation = irrep.D_from_angles(
            torch.tensor(sym.euler_angle[0], dtype = float), 
            torch.tensor(sym.euler_angle[1], dtype = float), 
            torch.tensor(sym.euler_angle[2], dtype = float), 
            torch.tensor(sym.inversion, dtype = float)
        ) 
        transformed = torch.einsum("ij,kj -> ki", rotation, values[:, permutation])
        return transformed[:, reverse]

def index_rotation(rotations):
    test = "a b c".split()
    for rot in rotations:
        rotated = ["", "", ""]
        for i in range(3):
            for j in range(3):
                if abs(rot[i][j]) < 1e-3:
                    continue
                rotated[i] += "{:>3d}{:s}".format( int(rot[i][j]), test[j] )
    print("{:s},{:s},{:s}".format(*rotated))

scf = SCFout("scf/scf.out")
cell = (scf.lattice, scf.positions, scf.types)


if __name__ == "__main__":
    dataset = spglib.get_symmetry_dataset(cell)
    rotations_crystal = dataset["rotations"]

    # build a reference table
    basis_functions = ["s px py pz".split()] * len(scf.positions)
    basis_functions_name = []
    basis_functions_index = {}
    for i, bfs in enumerate(basis_functions):
        for bf in bfs:
            name = i, bf
            basis_functions_index[name] = len(basis_functions_name)
            basis_functions_name.append(name)
    nbf = len(basis_functions_name)
    natom = len(dataset["std_positions"])
    # determine how they transform
    # since everything is direction is orthogonal, we can easily form a representation
    positions = deepcopy(dataset["std_positions"])
    positions -= positions[0]
    representation = np.zeros((len(rotations_crystal), nbf, nbf), dtype = float)

    basis_functions = "s px py pz".split()
    for irot, rot in enumerate(rotations_crystal):
        for ibf, bf in enumerate(basis_functions_name):
            iatom, f = bf
            v = positions[iatom]
            vb = np.zeros(4)
            vb[basis_functions.index(f)] += 1.0
                
            # rotated atoms 
            rv = rot.dot(v)
            # rotated basis function
            rot_bf = np.zeros((len(vb),len(vb)))
            rot_bf[0,0] = 1
            rot_bf[1:4, 1:4] = rot
            rvb = rot_bf.dot(vb)
            # rotated basis function
            for jatom in range(natom):
                if np.isclose(rv, positions[jatom]).all():
                    for bfs, value in zip(basis_functions, rvb):
                        jbf = basis_functions_index[(jatom, bfs)]
                        representation[irot, jbf, ibf] = value

    symmetrized_function = np.sum(representation, axis=0) / len(representation)
    print(symmetrized_function.shape)

    #print(positions)
    row_str = ""
    for bfn in basis_functions_name:
        row_str += "{:>4s}".format(str(bfn[0]) + bfn[1])
    print(row_str)

    for i, row in enumerate(symmetrized_function):
        row_str = ""
        for value in row:
            row_str += "{:>4.0f}".format(value)
        print(row_str)

    # it seems that we can find first the 1 dimensional basis
    # https://math.stackexchange.com/questions/1475212/simultaneously-diagonalize-the-regular-representation-of-c2-c2-c2

