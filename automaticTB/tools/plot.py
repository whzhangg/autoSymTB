import numpy as np
import typing

def get_cell_from_origin_centered_positions(positions: typing.List[np.ndarray]) -> np.ndarray:
    rmax = -1.0
    for pos in positions:
        rmax = max(rmax, np.linalg.norm(pos))
    cell = 4.0 * rmax * np.eye(3)
    return cell