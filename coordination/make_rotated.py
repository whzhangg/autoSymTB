from DFTtools.tools import read_generic_to_cpt, write_cif_file, atom_from_cpt
from e3nn import o3
import torch
import numpy as np
from copy import deepcopy
import os

#c, p, t = read_generic_to_cpt("ICSD_CollCode161491.cif")
#c_torch = torch.tensor(np.array(c), dtype = torch.float)

#print(c)

#print(o3.rand_matrix().type())
#print(c_torch.type())
#pos = torch.einsum('ij,zj->zi', o3.rand_matrix(), c_torch)
cell = "-4.52757000   4.52757000   4.52757000   4.52757000  -4.52757000   4.52757000   4.52757000   4.52757000  -4.52757000"
cell = np.array(cell.split(), dtype = float).reshape((3,3))
print(cell)

c_torch = torch.tensor(cell, dtype = torch.float)

with open("template/scf_template.in", 'r') as f:
    text = f.read()

folder = "rotated_ones"
for i in range(1, 21):
    pos = torch.einsum('ij,zj->zi', o3.rand_matrix(), c_torch)
    cell = pos.numpy()
    t = deepcopy(text)
    t += "CELL_PARAMETERS angstrom\n"
    for v in cell:
        t += "  {:>10.5f}{:>10.5f}{:>10.5f}\n".format(v[0],v[1],v[2])
    os.system("mkdir rotated_ones/rot{:0>d}".format(i))
    with open(os.path.join(folder, "rot{:0>d}".format(i), "scf.in"), 'w') as f:
        f.write(t)
    os.system("cp template/proj.in rotated_ones/rot{:0>d}".format(i))