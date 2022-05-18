import numpy as np


def find_RCL(cell: np.ndarray) -> np.ndarray:
    '''
    parameters: a1,a2,a3 the vector of lattice vector in unit of angstrom 
    return:     b1,b2,b3 the vector of reciprocal vector in 1/A
    '''
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]
    volumn=np.cross(a1,a2).dot(a3)
    b1=(2*np.pi/volumn)*np.cross(a2,a3)
    b2=(2*np.pi/volumn)*np.cross(a3,a1)
    b3=(2*np.pi/volumn)*np.cross(a1,a2)
    return np.vstack([b1,b2,b3])

def print_matrix(m: np.ndarray, format:str):
    # print an matrix using the given string format
    assert len(m.shape) == 2
    result = ""
    for row in m:
        for cell in row:
            result += " " + format.format(cell)
        result += "\n"
    print(result)

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