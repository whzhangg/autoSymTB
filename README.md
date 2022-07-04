# AutomaticTB
### Introduction
This package use symmetry adapted linear combination of atomic orbitals (SALCAOs) to construct 
a symmetrized tight-binding model, which can be solved to obtain the electronic structure if 
suitable parameters are supplied. It is intended to be used for machine learning (ML) projects 
where orbital interaction $\langle \phi_{\mu} | H | \phi_{\nv} \rangle$ will be the prediction 
target. 

### Methods
There are four core functionalities of this python module:
1. Character assignment for the symmetry operation in the site-symmetry group,
2. Formation of symmetrized linear combinations of atomic orbitals, and 
3. Handle symmetry relationship between different matrix elements in the Hamiltonian and overlap
4. Solution of electronic properties based on solving the Hamiltonian

### Usage
The codebase serve mainly as a library, but core functionalities can be called using prepared 
scripts 