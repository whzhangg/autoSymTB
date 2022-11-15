import numpy as np
import typing
from .pair import Pair

__all__ = [
    "tensor_dot",
    "random_rotation_matrix",
    "get_cell_from_origin_centered_positions"
]

def get_cell_from_origin_centered_positions(positions: typing.List[np.ndarray]) -> np.ndarray:
    rmax = -1.0
    for pos in positions:
        rmax = max(rmax, np.linalg.norm(pos))
    cell = 4.0 * rmax * np.eye(3)
    return cell

def tensor_dot(list_left, list_right) -> typing.List[Pair]:
    r = []
    for item1 in list_left:
        for item2 in list_right:
            r.append(Pair(item1, item2))
    return r


def random_rotation_matrix() -> np.ndarray:
    angles = np.random.rand(3) * 2 * np.pi
    sin = np.sin(angles)
    cos = np.cos(angles)
    yaw = np.array([[1,      0,       0],
                    [0, cos[0], -sin[0]],
                    [0, sin[0],  cos[0]]])
    pitch=np.array([[ cos[1], 0, sin[1]],
                    [0,       1,      0],
                    [-sin[1], 0, cos[1]]])
    roll = np.array([[cos[2],-sin[2], 0],
                    [sin[2], cos[2], 0],
                    [0,           0, 1]])
    return yaw.dot(pitch).dot(roll)


