from symmetrize import AORepresentation, cell, test_o3_basisfunction, rotations_D3h, D3h_positions

position = cell[1]
position[:] -= position[0]
rep = AORepresentation(D3h_positions, lmax = 1, symmetries= rotations_D3h)