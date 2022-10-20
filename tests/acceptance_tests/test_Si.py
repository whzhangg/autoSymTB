from automaticTB.examples.structures import get_Si_structure_2s2p3s
from automaticTB.functions import (
    get_namedLCs_from_nncluster,
    get_free_AOpairs_from_nncluster_and_namedLCs
)

si_2s2p3s_irreps = {
    "A_1 @ 1^th A_1",
    "A_1 @ 2^th A_1",
    "A_1 @ 3^th A_1",
    "A_1 @ 4^th A_1",
    "A_1 @ 5^th A_1",
    "E->A_1 @ 1^th E",
    "E->A_2 @ 1^th E",
    "T_1->A_2 @ 1^th T_1",
    "T_1->B_1 @ 1^th T_1",
    "T_1->B_2 @ 1^th T_1",
    "T_2->A_1 @ 1^th T_2",
    "T_2->B_1 @ 1^th T_2",
    "T_2->B_2 @ 1^th T_2",
    "T_2->A_1 @ 2^th T_2",
    "T_2->B_1 @ 2^th T_2",
    "T_2->B_2 @ 2^th T_2",
    "T_2->A_1 @ 3^th T_2",
    "T_2->B_1 @ 3^th T_2",
    "T_2->B_2 @ 3^th T_2",
    "T_2->A_1 @ 4^th T_2",
    "T_2->B_1 @ 4^th T_2",
    "T_2->B_2 @ 4^th T_2",
    "T_2->A_1 @ 5^th T_2",
    "T_2->B_1 @ 5^th T_2",
    "T_2->B_2 @ 5^th T_2"
}

def test_Si_decomposition():
    bulksi = get_Si_structure_2s2p3s()
    si_environment = bulksi.nnclusters[0]
    named_lcs = get_namedLCs_from_nncluster(si_environment)
    
    assert len(named_lcs) == si_environment.orbitalslist.num_orb
    for nlc in named_lcs:
        assert str(nlc.name) in si_2s2p3s_irreps
        
    free_pairs = get_free_AOpairs_from_nncluster_and_namedLCs(si_environment, named_lcs)

    assert len(free_pairs) == 15