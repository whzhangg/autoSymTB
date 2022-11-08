from automaticTB.examples import get_Si_structure
from automaticTB.functions import (
    get_equationsystem_from_structure_orbital_dict,
    get_tbModel_from_structure_interactions_overlaps
)
import numpy as np

bulksi = get_Si_structure()
si_orbital = {"Si":"3s3p"}


def test_Si_decomposition_sp():
    equation_system = get_equationsystem_from_structure_orbital_dict(bulksi, {"Si":"3s3p"})
    for i, free in enumerate(equation_system.free_AOpairs):
        print(f"{i+1:>2d}" + str(free))

    assert len(equation_system.free_AOpairs) == 9


def test_Si_decomposition_sps():
    equation_system = get_equationsystem_from_structure_orbital_dict(bulksi, {"Si":"3s3p4s"})
    for i, free in enumerate(equation_system.free_AOpairs):
        print(f"{i+1:>2d}" + str(free))

    assert len(equation_system.free_AOpairs) == 17


def test_Si_solve_dispersion_sps():
    combined = get_equationsystem_from_structure_orbital_dict(bulksi, {"Si":"3s3p4s"})

    values = np.array([
        -3.3179,    # 1 > Pair: Si-00      3s -> Si-00      3s @ (  0.00,  0.00,  0.00)
        8.23164,    # 2 > Pair: Si-01      4s -> Si-01      4s @ (  0.00,  0.00,  0.00)
        0.00000,    # 3 > Pair: Si-01      4s -> Si-00      3s @ (  1.36, -1.36, -1.36)
        0.00000,    # 4 > Pair: Si-00      3s -> Si-00      4s @ (  0.00,  0.00,  0.00)
        -9.5990/4,  # 5 > Pair: Si-01      3s -> Si-00      3s @ (  1.36, -1.36, -1.36)
        7.2505/4,   # 6 > Pair: Si-01      4s -> Si-00     3px @ (  1.36, -1.36, -1.36)
        1.67862,    # 7 > Pair: Si-00     3px -> Si-00     3px @ (  0.00,  0.00,  0.00)
        0.00000,    # 8 > Pair: Si-01      4s -> Si-00      4s @ (  1.36, -1.36, -1.36)
        7.1423/4,   # 9 > Pair: Si-01      3s -> Si-00     3px @ (  1.36, -1.36, -1.36)
        8.23164,    #10 > Pair: Si-00      4s -> Si-00      4s @ (  0.00,  0.00,  0.00)
        0.00000,    #11 > Pair: Si-01      3s -> Si-00      4s @ (  1.36, -1.36, -1.36)
        -3.3179,    #12 > Pair: Si-01      3s -> Si-01      3s @ (  0.00,  0.00,  0.00)
        -4.7757/4,  #13 > Pair: Si-01     3px -> Si-00     3pz @ (  1.36, -1.36, -1.36)
        1.6955/4,   #14 > Pair: Si-01     3px -> Si-00     3px @ (  1.36, -1.36, -1.36)
        -7.2505/4,  #15 > Pair: Si-01     3px -> Si-00      4s @ (  1.36, -1.36, -1.36)
        0.00000,    #16 > Pair: Si-01      3s -> Si-01      4s @ (  0.00,  0.00,  0.00)
        -7.1423/4,  #17 > Pair: Si-01     3px -> Si-00      3s @ (  1.36, -1.36, -1.36)
        1.67862,    #18 > Pair: Si-01     3px -> Si-01     3px @ (  0.00,  0.00,  0.00)
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

test_Si_solve_dispersion_sps()