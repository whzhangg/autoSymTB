from automaticTB.SALCs import NamedLC
from automaticTB.utilities import Pair
import typing
import numpy as np

def get_random_MO_interaction(mopairs: typing.List[Pair]) -> np.ndarray:
    stored = {}
    values = []
    for pair in mopairs:
        left: NamedLC = pair.left
        right: NamedLC = pair.right
        if left.name!=right.name:
            values.append(0)
        else:
            values.append(np.random.random())
    
    return np.array(values)