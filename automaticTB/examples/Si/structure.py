import os
from ...structure import Structure
from ...config import rootfolder
from ...tools import read_cif_to_cpt

structure_file = os.path.join(rootfolder, "examples", "Si", "Si_ICSD51688.cif")

def get_si_structure() -> Structure:
    c, p, t = read_cif_to_cpt("Si_ICSD51688.cif")
    # initially, I thought reading yml file is fast, but it turned out that 
    # creating everything from scratch is as fast as reading the yml file. ~ 1.7 s
    return Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.0)
