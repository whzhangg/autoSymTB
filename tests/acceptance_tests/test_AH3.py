from automaticTB.functions import get_namedLCs_from_nncluster
from automaticTB.examples.structures import get_AH3_nncluster
from automaticTB.functions import get_free_AOpairs_from_nncluster_and_namedLCs

def test_AH3_decomposition():
    ah3_irreps = [
        "A_''_2 @ 1^th A_''_2",
        "A_'_1 @ 1^th A_'_1",
        "A_'_1 @ 2^th A_'_1",
        "E_'->A_1 @ 1^th E_'",
        "E_'->B_2 @ 1^th E_'",
        "E_'->A_1 @ 2^th E_'",
        "E_'->B_2 @ 2^th E_'"
    ]
    ah3 = get_AH3_nncluster()
    named_lcs = get_namedLCs_from_nncluster(ah3)
    assert len(named_lcs) == ah3.orbitalslist.num_orb
    for nlc in named_lcs:
        assert str(nlc.name) in ah3_irreps

    free = get_free_AOpairs_from_nncluster_and_namedLCs(
        ah3, named_lcs
    )
    assert len(free) == 4