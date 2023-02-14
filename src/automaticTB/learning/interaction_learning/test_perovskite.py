from automaticTB.functions import (
    get_namedLCs_from_nncluster,
    get_free_AOpairs_from_nncluster_and_namedLCs
)
from automaticTB.examples.structures import get_perovskite_structure
from automaticTB.learning.orbital_interaction_adopter import (
    get_elementEncoder_orbitalEncoder_from_structure, 
    get_interaction_orbitals_from_structure_and_AOpair
)
from automaticTB.learning.interaction_mapper import get_nearest_neighbor_orbital_mapper
from automaticTB.printing import print_ao_pairs

structure = get_perovskite_structure()
ele_encoder, orb_encoder = get_elementEncoder_orbitalEncoder_from_structure(
    structure
)

mapper = get_nearest_neighbor_orbital_mapper(
    len(ele_encoder.atomic_number_set), 
    orb_encoder.irreps_str
)

for i, nncluster in enumerate(structure.nnclusters):
    print(nncluster.sitesymmetrygroup.groupname)
    named_lcs = get_namedLCs_from_nncluster(nncluster)
    free_pairs = get_free_AOpairs_from_nncluster_and_namedLCs(nncluster, named_lcs)
    #print_ao_pairs(nncluster, free_pairs)
    for pair in free_pairs:
        geometry_data = get_interaction_orbitals_from_structure_and_AOpair(
            structure, pair
        ).geometry_data
        print(geometry_data.pos)
        print(mapper(geometry_data))