import numpy as np
import dataclasses, typing

from automaticTB.examples import get_perovskite_structure
from automaticTB.examples.Perovskite.ao_interaction import (
    get_interaction_values_from_list_AOpairs
)

from automaticTB.functions import (
    get_equationsystem_from_structure_orbital_dict,
    get_tbModel_from_structure_interactions_overlaps
)


structure = get_perovskite_structure()
combined = get_equationsystem_from_structure_orbital_dict(
    structure, {"Pb":"6s6p", "Cl": "3s3p"})

          
def test_solve_energy():
    free_ao = combined.free_AOpairs

    free_values \
        = get_interaction_values_from_list_AOpairs(structure.cell, structure.positions, free_ao)

    solved = combined.solve_interactions_to_InteractionPairs(free_values)

    model = get_tbModel_from_structure_interactions_overlaps(
        structure, solved
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
