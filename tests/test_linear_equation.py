import numpy as np
from automaticTB.tools import find_free_variable_indices_by_row_echelon, LinearEquation

def prepare_tests(n_tot: int, n_free: int):
    """
    prepare a sample question, ao_interaction and transformation between AO interaction and MO 
    interaction are random. we substract the final row from each of the first n_free row to obtain
    a homogeneous equaton. 
    """
    transformation = (np.random.rand(n_tot, n_tot) - 0.5)
    ao_interaction = np.random.rand(n_tot)
    mo_interaction = transformation @ ao_interaction

    for i in range(0, n_tot - n_free): 
        ratio = (-1 * mo_interaction[i]) / mo_interaction[-1] 
        transformation[i] += ratio * transformation[-1]
        mo_interaction[i] += ratio * mo_interaction[-1]

    homogeneous_equation = transformation[0:n_tot - n_free]
    free_variable_index = find_free_variable_indices_by_row_echelon(homogeneous_equation)

    some_answers = np.array([ao_interaction[fvi] for fvi in free_variable_index])
    return homogeneous_equation, some_answers, ao_interaction


def test_homogeneous_equation_solver():
    equation, some_answers, all_answers = prepare_tests(20, 5)
    equation = LinearEquation(equation)
    solution = equation.solve_providing_values(some_answers)
    assert np.allclose(solution, all_answers)
    