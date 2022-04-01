from DFTtools.tools import read_generic_to_ase_atoms

atoms = read_generic_to_ase_atoms("SF6_mp.cif")
positions = atoms.get_positions()
print(positions)

S = positions[0]
F = positions[1:]
delta = F - S
print(delta)
a = 10.0

import numpy as np
S = np.array([5.0,5.0,5.0])
F = S + delta

pos = np.vstack([S,F]) / 10.0

for p, sym in zip(pos, atoms.get_chemical_symbols()):
    print("{:>2s}  {:>10.6f}{:>10.6f}{:>10.6f}".format(sym, *p))

