import spglib
from DFTtools.tools import read_generic_to_cpt

c,p,t = read_generic_to_cpt("/Users/wenhao/work/code/python/DFTtools/tests/230space_group/187_wc.cif")
sym_data = spglib.get_symmetry_dataset((c,p,t))

syms = sym_data["rotations"]
for i, rot in enumerate(syms):
    print(i)
    print(rot)

# in this case, we have the result 
# E, S3, C3, sigma_h, C3, S3, C2', sigma_v, C2', sigma_h, C2', sigma_h