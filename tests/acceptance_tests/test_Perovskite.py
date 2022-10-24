from automaticTB.functions import (
    get_namedLCs_from_nncluster, 
    get_free_AOpairs_from_nncluster_and_namedLCs
)
from automaticTB.interaction import InteractionEquation, CombinedInteractionEquation
from automaticTB.examples.Perovskite.ao_interaction import (
    get_interaction_values_from_list_AOpairs
)
import numpy as np
from automaticTB.examples.structures import get_perovskite_structure
import dataclasses, typing
from automaticTB.printing import print_ao_pairs
from automaticTB.tightbinding import gather_InteractionPairs_into_HijRs, TightBindingModel

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
        7, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u"
        }
    ),
    Answer(
        7, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u",
        }
    ),
    Answer(
        7, "4/mmm",
        irreps = {
            "A_1g @ 1^th A_1g","A_1g @ 2^th A_1g","A_1g @ 3^th A_1g","A_2u @ 1^th A_2u",
            "A_2u @ 2^th A_2u","A_2u @ 3^th A_2u","E_g->B_1 @ 1^th E_g","E_g->B_2 @ 1^th E_g",
            "E_u->B_1 @ 1^th E_u","E_u->B_2 @ 1^th E_u","E_u->B_1 @ 2^th E_u",
            "E_u->B_2 @ 2^th E_u",          
        }
    )
]

structure = get_perovskite_structure()
all_named_lcs = []
for nncluster in structure.nnclusters:
    all_named_lcs.append(get_namedLCs_from_nncluster(nncluster))

def test_perovskite_decomposition():
    for answer, nncluster, named_lcs in zip(answers, structure.nnclusters, all_named_lcs):
        assert nncluster.sitesymmetrygroup.groupname == answer.symmetry
        assert len(named_lcs) == nncluster.orbitalslist.num_orb
        for nlc in named_lcs:
            assert str(nlc.name) in answer.irreps       
        free = get_free_AOpairs_from_nncluster_and_namedLCs(nncluster, named_lcs)
        assert len(free) == answer.nfree

def test_perovskite_combined_free_parameter():
    equation_systems = []
    for nncluster, named_lcs in zip(structure.nnclusters, all_named_lcs):
        equation_systems.append(
            InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        )
    combined = CombinedInteractionEquation(equation_systems)
    assert len(combined.free_AOpairs) == 9
    
def test_recover_values():
    equation_systems = []
    for nncluster, named_lcs in zip(structure.nnclusters, all_named_lcs):
        equation_systems.append(
            InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        )
    combined = CombinedInteractionEquation(equation_systems)
    free_ao = combined.free_AOpairs

    free_values \
        = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_ao)

    solved = combined.solve_interactions_to_InteractionPairs(free_values)
    for aopair_with_value in solved:
        correct_value = get_interaction_values_from_list_AOpairs(
            structure.cell, structure.positions, [aopair_with_value]
        )[0]
        assert np.isclose(aopair_with_value.value.real, correct_value, atol=1e-3)
            
def test_solve_energy():
    equation_systems = []
    for nncluster, named_lcs in zip(structure.nnclusters, all_named_lcs):
        equation_systems.append(
            InteractionEquation.from_nncluster_namedLC(nncluster, named_lcs)
        )
    combined = CombinedInteractionEquation(equation_systems)
    free_ao = combined.free_AOpairs

    free_values \
        = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_ao)

    solved = combined.solve_interactions_to_InteractionPairs(free_values)
    HijRs = gather_InteractionPairs_into_HijRs(solved)
    model = TightBindingModel(
        structure.cell, structure.positions, structure.types, HijRs
    )
    cubic_kpos_results = {
        "M": (  np.array([0.5,0.5,0.0]),
                np.array([
                    -14.37777,-13.30312,-13.30312, -9.56827, -7.24541, -2.65415, 
                     -2.65415, -2.16000, -2.16000, -2.16000, -2.16000, -2.16000, 
                     -0.63397,  2.97728,  2.97728,  7.47541
                ])
            ),
        "X": (  np.array([0.5,0.0,0.0]),
                np.array([
                    -15.02604,-13.30452,-13.21000, -8.27217, -7.41731, -7.41731,
                     -3.07340, -2.16000, -2.16000, -2.16000, -2.16000, -2.16000, 
                     -1.28178,  3.39791,  7.64731,  7.64731
                ])
            ),
        "G": (  np.array([0.0,0.0,0.0]),
                np.array([
                    -15.51349,-13.21000,-13.21000, -7.58537, -7.58537, -7.58536, 
                     -6.90651, -2.16000, -2.16000, -2.16000, -2.16000, -2.16000, 
                     -2.16000,  7.81536,  7.81537,  7.81537
                ])
            ),
        "R": (  np.array([0.5,0.5,0.5]),
                np.array([
                    -13.30177,-13.30177,-13.30177,-11.26851, -2.16000, -2.16000, 
                     -2.16000, -2.16000, -2.16000, -2.16000, -2.16000, -2.16000, 
                     -0.10149,  2.48177,  2.48177,  2.48177
                ])
            ),
    }

    for value in cubic_kpos_results.values():
        kpos, result = value
        e, _ = model.solveE_at_k(kpos)
        assert np.allclose(e, result, atol = 1e-4)
