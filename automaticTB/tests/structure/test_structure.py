from automaticTB.structure.structure import Structure
from DFTtools.tools import read_generic_to_cpt
from DFTtools.config import rootfolder
import os

Si_cif_fn = os.path.join(rootfolder, "tests/structures/Si_ICSD51688.cif")

def test_load_structure_cutoff():
    from DFTtools.tools import read_generic_to_cpt
    c,p,t = read_generic_to_cpt(Si_cif_fn)
    structure: Structure = Structure.from_cpt_rcut(c, p, t, 3.0)

    assert len(structure.nnclusters) == 2
    assert len(structure.cartesian_rotations) == 48

    first_cluster = structure.nnclusters[0]

    group = \
        first_cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
    assert group.groupname == "-43m"

    equivalent_atoms = first_cluster.get_equivalent_atoms_from_SiteSymmetryGroup(group)
    assert equivalent_atoms == {0: {0,1,2,3}, 4: {4}}