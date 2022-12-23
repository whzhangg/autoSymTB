"""
this module depends on the solve module and can be considered to be an extension of solve, 
before handling the data to the module "supercell"

when considering doped supercell, we only need to solve additional perturbation if we have
additional orbitals
"""

from .make_supercell import StructureAndEquation