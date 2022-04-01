import spglib
from DFTtools.DFT.output_parser import SCFout
import numpy as np
from copy import deepcopy

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

