import typing

import numpy as np
from ._analysis_params import calc_2order

def generate_perturbation_rand(
        deltas: np.ndarray, nsample: int, random_seed: int = None) -> np.ndarray:
    """generate 0 centered nsample data, first one is zeros
    
    For each value, the returned range is [-delta, delta]
    """
    nsf = len(deltas)
    first = np.zeros_like(deltas)
    if nsample == 1:
        return np.array([first])
    
    if not random_seed:
        random_seed = int(np.random.rand() * 1e5)

    rng = np.random.default_rng(random_seed)
    
    dx = (rng.random((nsample-1, nsf)) - 0.5) * 2
    dx = dx.dot(np.diag(deltas))
    
    return np.vstack([first, dx])


def generate_perturbation_sobol(
    names_and_deltas: typing.Dict[str, float], nsample: int, random_seed: int = None
) -> typing.Tuple[dict, np.ndarray]:
    """return sobol problem and perturbation

    The actual number of samples is nsample * (2 + nparams)
    
    Parameters
    ----------
    names_and_deltas: dict[str, float]
        parameter names and their range as [-delta, delta]
    nsample: int
        nsample for salib

    Note: perturbations are zero centered
    """
    from SALib.sample.sobol import sample
    names = [k for k,_ in names_and_deltas.items()]
    delta = [v for _,v in names_and_deltas.items()]
    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': [[-1 * d, d] for d in delta]
        }
    
    if not random_seed:
        random_seed = int(np.random.rand() * 1e5)

    param_values = sample(
        problem, nsample, calc_second_order=calc_2order, seed=random_seed)
                
    return problem, param_values
    
