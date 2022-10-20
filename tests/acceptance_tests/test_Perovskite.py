from automaticTB.functions import (
    get_namedLCs_from_nncluster, 
    get_free_AOpairs_from_nncluster_and_namedLCs
)

from automaticTB.examples.structures import get_perovskite_structure
import dataclasses, typing

@dataclasses.dataclass
class Answer:
    nfree : int
    symmetry: str
    irreps: typing.List[str]

answers = [
    Answer(
        7, "m-3m", 
        irreps = {
            "A_1g @ 1^th A_1g", "A_1g @ 2^th A_1g", "A_1g @ 3^th A_1g", "E_g->A_1 @ 1^th E_g",
            "E_g->A_2 @ 1^th E_g","E_g->A_1 @ 2^th E_g","E_g->A_2 @ 2^th E_g",
            "T_1g->A_2 @ 1^th T_1g","T_1g->B_1 @ 1^th T_1g","T_1g->B_2 @ 1^th T_1g",
            "T_1u->A_1 @ 1^th T_1u","T_1u->B_1 @ 1^th T_1u","T_1u->B_2 @ 1^th T_1u",
            "T_1u->A_1 @ 2^th T_1u","T_1u->B_1 @ 2^th T_1u","T_1u->B_2 @ 2^th T_1u",
            "T_1u->A_1 @ 3^th T_1u","T_1u->B_1 @ 3^th T_1u","T_1u->B_2 @ 3^th T_1u",
            "T_1u->A_1 @ 4^th T_1u","T_1u->B_1 @ 4^th T_1u","T_1u->B_2 @ 4^th T_1u",
            "T_2g->A_1 @ 1^th T_2g","T_2g->B_1 @ 1^th T_2g","T_2g->B_2 @ 1^th T_2g",
            "T_2u->A_2 @ 1^th T_2u","T_2u->B_1 @ 1^th T_2u","T_2u->B_2 @ 1^th T_2u"
        }
    ),
    Answer(
        8, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u"
        }
    ),
    Answer(
        8, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u",
        }
    ),
    Answer(
        8, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u",          
        }
    )
]


def test_perovskite_decomposition():
    structure = get_perovskite_structure()
    for answer, nncluster in zip(answers, structure.nnclusters):
        assert nncluster.sitesymmetrygroup.groupname == answer.symmetry

        named_lcs = get_namedLCs_from_nncluster(nncluster)
        assert len(named_lcs) == nncluster.orbitalslist.num_orb
        for nlc in named_lcs:
            assert str(nlc.name) in answer.irreps
        
        free = get_free_AOpairs_from_nncluster_and_namedLCs(nncluster, named_lcs)
        assert len(free) == answer.nfree
