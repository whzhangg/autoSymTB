import typing
import numpy as np

# this module store group multiplication table as a 
# set of GroupMultiplication, in the form of g1 * g2 -> h
# where g1, g2 and h are seitz symbol of the group

class GroupMultiplication(typing.NamedTuple):
    g1: str
    g2: str
    h: str

def get_multiplication_table(seitz_operation: typing.Dict[str, np.ndarray]):
    table: typing.Set[GroupMultiplication] = set()
    for seitz1, op1 in seitz_operation.items():
        for seitz2, op2 in seitz_operation.items():
            matrix = np.matmul(op1, op2)
            for seitz3, op3 in seitz_operation.items():
                if np.allclose(matrix, op3):
                    table.add(
                        GroupMultiplication(seitz1, seitz2, seitz3)
                    )
                    break
    
    assert len(table) == len(seitz_operation) **2
    return table

