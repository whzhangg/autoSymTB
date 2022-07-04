"""
this package provide the functionality to generate the linear combination 
of the atomic orbitals into MOs

the input would be a crystal structure
For each atoms:
1. We find its site symmetry
2. We find its neighbors
3. We obtain the SALC for the atomic orbitals on the atom
4. We obtain the SALC for the atomic orbitals on its neighbors
5. the interactions are calculated using SALCs if allowed
6. A linear system of equations can be written down and solved
7. After the interaction is known in real space, we formulate the Hamiltion at different k

I will work on the numpy version, the rotation matrix is in the basis of:
https://xaktly.com/Chemistry_Electrons.html
y_0_0, y_1_-1, y_1_0, y_1_1, y_2_-2, y_2_-1, y_2_0,  y_2_1, y_2_2
    s,     px,    py,    pz,    dxy,    dxz,   dyz, dx2-y2,   dz2


"""