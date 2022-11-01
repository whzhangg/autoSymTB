from automaticTB.examples import get_Si_structure_2s2p3s
from automaticTB.functions import (
    get_namedLCs_from_nncluster,
    get_free_AOpairs_from_nncluster_and_namedLCs,
    get_tbModel_from_structure_interactions_overlaps,
    get_combined_equation_from_structure
)
import numpy as np

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

system_free_interaction = [
"AOPair(l_AO=i=0(Si) n=3 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=3 l=0 m=0 @ 0.00 0.00 0.00)",
"AOPair(l_AO=i=0(Si) n=2 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=0 m=0 @ 0.00 0.00 0.00)",
"AOPair(l_AO=i=0(Si) n=2 l=1 m=-1 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=1 m=-1 @ 0.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=3 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=0 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=3 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=1 m=1 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=0 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=3 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=3 l=0 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=1 m=1 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=0 m=0 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=3 l=0 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=1 m=1 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=0 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=1 m=1 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=1 m=0 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=1 m=1 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=2 l=1 m=1 @-1.00 0.00 0.00)",
"AOPair(l_AO=i=1(Si) n=2 l=1 m=1 @ 0.00 0.00 0.00, r_AO=i=0(Si) n=3 l=0 m=0 @-1.00 0.00 0.00)",
]

bulksi = get_Si_structure_2s2p3s()
all_named_lcs = []
for si_environment in bulksi.nnclusters:
    all_named_lcs.append(get_namedLCs_from_nncluster(si_environment))

def test_Si_decomposition():
    named_lcs = all_named_lcs[0]
    si_environment = bulksi.nnclusters[0]
    assert len(named_lcs) == si_environment.orbitalslist.num_orb
    for nlc in named_lcs:
        assert str(nlc.name) in si_2s2p3s_irreps
        
    free_pairs = get_free_AOpairs_from_nncluster_and_namedLCs(si_environment, named_lcs)

    assert len(free_pairs) == 13

def test_Si_solve_dispersion():
    combined = get_combined_equation_from_structure(bulksi)

    for aopair, answer in zip(combined.free_AOpairs, system_free_interaction):
        assert str(aopair) == answer

    values = np.array([
        8.23164, -3.3179, 1.67862, 0.0, 7.2505/4, -9.599/4, 0.0, 
        7.1423/4, 0.0, -7.1423/4, -4.7757/4, 1.6955/4, -7.2505/4
    ])

    solved_interaction = combined.solve_interactions_to_InteractionPairs(values)
    #HijRs = gather_InteractionPairs_into_HijRs(solved_interaction)
    model = get_tbModel_from_structure_interactions_overlaps(
        bulksi, solved_interaction
    )

    kpos_results = {
        "L": (  np.array([0.5,0.5,0.5]),
                np.array([
                    -10.78870,  -8.45989,  -1.55698,  -1.55698,   2.43320,
                      3.95976,   4.91422,   4.91422,  11.96404,  14.07631
                ])
            ),
        "G": (  np.array([0.0,0.0,0.0]),
                np.array([
                    -12.91690,  -0.01688,  -0.01688,  -0.01688,   3.37412,
                      3.37412,   3.37412,   6.28110,   8.23164,   8.23164
                ])
            ),
        "X": (  np.array([0.5,0.0,0.5]),
                np.array([
                    -9.51511,  -9.51511,  -3.09708,  -3.09708,   2.20191,
                     2.20191,   6.45432,   6.45432,  13.90557,  13.90557
                ])
            ),
        "K": (  np.array([3/8,3/8,3/4]),
                np.array([
                    -9.90186,  -9.13006,  -3.32185,  -2.64599,   1.94085,
                     2.60969,   6.00323,   6.67957,  13.59383,  14.07179
                ])
            ),
    }

    for value in kpos_results.values():
        kpos, result = value
        e, _ = model.solveE_at_k(kpos)
        assert np.allclose(e, result, atol = 1e-4)
