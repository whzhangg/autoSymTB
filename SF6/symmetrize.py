import spglib
from DFTtools.DFT.output_parser import SCFout
import numpy as np
from copy import deepcopy
from e3nn.o3 import Irreps
from numpy import pi
import torch
from dataclasses import dataclass
from typing import Tuple, List, Union, Any

# dataclass generate initialization automatically
@dataclass
class Symmetry:
    euler_angle : Tuple[float]
    inversion : int


def find_symmetry_spglib(struct: tuple) -> list:
    # return rotation in cartesian coordinates
    dataset = spglib.get_symmetry_dataset(struct)
    rotations_crystal = dataset["rotations"]
    cellT = struct[0].T
    rotations_cartesian = [ cellT.dot(r).dot(np.linalg.inv(cellT)) for r in rotations_crystal ]
    return rotations_cartesian


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


def apply_symmetry_operation(
        sym: Symmetry, irreps: Union[str, Irreps], values: torch.tensor
    ) -> torch.tensor:
    repres = Irreps(irreps)
    assert repres.dim == values.shape[-1]
    rotation = repres.D_from_angles(
            torch.tensor(sym.euler_angle[0], dtype = float), 
            torch.tensor(sym.euler_angle[1], dtype = float), 
            torch.tensor(sym.euler_angle[2], dtype = float), 
            torch.tensor(sym.inversion, dtype = float)
        ) 
    return torch.einsum("ij,kj -> ki", rotation, values)

# AO representation
class AORepresentation:
    """
    passing a tuple and a maximum l, we construct 
    the representation in the atomic orbital vector space
    with dimension natom * nbasis, 
    we define the sequence of the atomic orbital following e3nn
    s x z y xy xz z2 zy x2-y2
    and rotations (alpha, beta, gamma) to be around z, x, z axis that do not rotate with the body
    """
    defined_orbitals = "s x z y xy xz z2 zy x2-y2".split()
    permute_yz = torch.tensor([0, 2, 1])
    def __init__(self, positions: Any, AOs: List[Union[Irreps, str]], symmetries: List[Symmetry]) -> None:
        positions = torch.tensor(positions, dtype = float)
        self.positions = torch.index_select(positions, 1, self.permute_yz)
        self.symmetry_operations = symmetries
        self.check_symmetry()

        self.nsym = len(self.symmetry_operations)
        self.natom = len(self.positions)
        assert len(AOs) == self.natom
        self.AOs = [Irreps(ir) for ir in AOs]
        self.bfs = [(iatom,j) for iatom in range(self.natom) for j in range(self.AOs[iatom].dim)]
        self.bf_index = { bf:i for i, bf in enumerate(self.bfs)}
        self.nbfs = len(self.bfs)
        self.generate_representation()

    def generate_representation(self):
        matrix = torch.zeros((self.nsym, self.nbfs, self.nbfs))
        for isym,sym in enumerate(self.symmetry_operations):
            rotated_positions = apply_symmetry_operation(sym, "1x1o", self.positions)
            for iatom, ipos in enumerate(self.positions):
                for jatom, jpos in enumerate(rotated_positions):
                    if torch.allclose(ipos, jpos):
                        # iatom is rotated to jatom
                        assert self.AOs[iatom] == self.AOs[jatom]
                        rotated_AOs = apply_symmetry_operation(sym, self.AOs[iatom], torch.eye(self.AOs[iatom].dim, dtype = float))
                        for iao in range(self.AOs[iatom].dim):
                            for jao in range(self.AOs[iatom].dim):
                                i_index = self.bf_index[(iatom,iao)]
                                j_index = self.bf_index[(jatom,jao)]
                                matrix[isym, j_index, i_index] = rotated_AOs[iao, jao] # equation 4.5
        traces = [torch.trace(m) for m in matrix]
        print(traces) 
                        

    def check_symmetry(self):
        # check if symmetry is indeed the same
        #cartesian_positions = np.einsum("ij, kj -> ki", self.cpt[0],self.cpt[1])
        # we use strings to compare positions
        cartesian_positions = self.positions
        pos_list = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in cartesian_positions.detach()])
        for symmetry in self.symmetry_operations:
            transformed = apply_symmetry_operation(symmetry, "1x1o", cartesian_positions)
            transformed = set(["{:>7.3f},{:>7.3f},{:>7.3f}".format(*(p+0.000001)) for p in transformed.detach()])
            assert transformed == pos_list, (transformed, symmetry)


if __name__ == "__main__":
    scf = SCFout("scf/scf.out")
    cell = (scf.lattice, scf.positions, scf.types)
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
    