from symmetrize import AORepresentation, Symmetry
import numpy as np

rotations_D3h = [
    Symmetry((0,0,0), 0), # identity
    Symmetry((0,0,np.pi), 1), # horizontal refection
    Symmetry((0,0,2*np.pi/3), 0), # 120degree rotation
    Symmetry((0,0,  np.pi/3), 1), # S3
    Symmetry((0,np.pi,0), 0), # C2: rotation around x 
    Symmetry((0,np.pi,-np.pi), 1) # vertical ?
]

D3h_positions = [
    [ 0.0,   0.0, 0.0], 
    [ 2.0,   0.0, 0.0],
    [-1.0, np.sqrt(3), 0.0],
    [-1.0,-np.sqrt(3), 0.0],
]
AOs = [ '1x0e + 1x1o' for pos in D3h_positions ]

rep = AORepresentation(D3h_positions, AOs, rotations_D3h)