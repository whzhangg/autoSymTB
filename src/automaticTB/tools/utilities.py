import typing, numpy as np

from automaticTB.parameters import zero_tolerance


def get_cell_from_origin_centered_positions(positions: typing.List[np.ndarray]) -> np.ndarray:
    rmax = -1.0
    for pos in positions:
        rmax = max(rmax, np.linalg.norm(pos))
    cell = 4.0 * rmax * np.eye(3)
    return cell


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


def plot_value_distribution(data: np.ndarray, filename : str) -> None:
    """write to file (.dat/.pdf) the distribution of values in an array"""
    flattened = np.sort(np.abs(data.flatten()))
    if filename.endswith(".dat"):
        with open(filename, 'w') as f:
            for i, d in enumerate(flattened):
                f.write(f"{i+1:>10d} {d:>12.6e}")
    else:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.add_subplot()
        axes.set_xlabel("Num. points")
        axes.set_ylabel("Values abs()")
        maximum = np.max(flattened) * 10
        min_range = zero_tolerance ** 2 / maximum
        axes.set_ylim(min_range, maximum)
        axes.set_yscale('log')
        x = np.arange(len(flattened))
        axes.plot(x, flattened, 'o')
        fig.savefig(filename)


def print_np_matrix(m: np.ndarray, format:str = "{:>6.2f}"):
    # print an matrix using the given string format
    assert len(m.shape) == 2
    result = ""
    for row in m:
        for cell in row:
            result += " " + format.format(cell)
        result += "\n"
    print(result)