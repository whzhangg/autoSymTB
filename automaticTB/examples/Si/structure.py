from automaticTB.structure import Structure
from automaticTB.utilities import load_yaml, save_yaml
import os

structure_file = "Si_structure.yml"

if os.path.exists(structure_file):
    Si_structure = load_yaml(structure_file)
else:
    from DFTtools.tools import read_generic_to_cpt
    c, p, t = read_generic_to_cpt("Si_ICSD51688.cif")
    Si_structure = Structure.from_cpt_rcut(c,p,t, {"Si": [0,1]}, 3.0)
    save_yaml(Si_structure,"Si_structure.yml")
