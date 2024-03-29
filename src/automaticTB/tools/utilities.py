import typing
import functools
import time

import numpy as np


def rotation_fraction_to_cartesian(input_rotations: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """accept single rotation as (3,3) or rotations as (n,3,3)"""
    cellT = cell.transpose()
    cellT_inv = np.linalg.inv(cellT)
    if input_rotations.shape == (3,3):
        step1 = np.matmul(cellT, input_rotations)
        return np.matmul(step1, cellT_inv)
    else:
        step1 = np.einsum("ij, njk -> nik", cellT, input_rotations)
        return np.einsum("nik, kj -> nij", step1, cellT_inv)


def rotation_cartesian_to_fraction(input_rotations: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """accept single rotation as (3,3) or rotations as (n,3,3)"""
    cellT = cell.transpose()
    cellT_inv = np.linalg.inv(cellT)
    if input_rotations.shape == (3,3):
        step1 = np.matmul(cellT_inv, input_rotations)
        return np.matmul(step1, cellT)
    else:
        step1 = np.einsum("ij, njk -> nik", cellT_inv, input_rotations)
        return np.einsum("nik, kj -> nij", step1, cellT)


def timefn(fn):
    """print the execution time in second for a given function"""
    @functools.wraps(fn)
    
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__:<40s} took {t2 - t1:>.8f} seconds")
        return result

    return measure_time


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
        min_range = 1e-10 ** 2 / maximum
        axes.set_ylim(min_range, maximum)
        axes.set_yscale('log')
        x = np.arange(len(flattened))
        axes.plot(x, flattened, 'o')
        fig.savefig(filename)


def print_np_matrix(m: np.ndarray, format_:str = "{:>6.3f}"):
    # print an matrix using the given string format
    assert len(m.shape) == 2
    result = ""
    for row in m:
        for cell in row:
            result += " " + format_.format(cell)
        result += "\n"
    print(result)

def print_np_matrix_complex(m: np.ndarray, format_: str = "{:>6.3f}"):
    import re
    length = int(re.findall(r'\d+', format_)[0])


    nbasis, nbasis = m.shape
    for i in range(nbasis):
        for j in range(nbasis):
            a = m[i,j].real
            b = m[i,j].imag
            if np.abs(m[i,j]) < 1e-4:
                print("[" + " "*(length*2+1) + "]", end=" ")
            else:
                formatter = f"[{format_},{format_}]"
                print("[{:>6.3f},{:>6.3f}]".format(a+1e-6,b+1e-6), end=" ")
        print("")