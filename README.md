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
The codebase serve mainly as a library, but core functionalities can be called using prepared scripts 

# Installation
1. Create an conda environment:
```
conda create --name autoTB python=3.8
conda activate autoTB
```
2. We need to add necessary channels
```
conda config --add channels pyg pytorch conda-forge
```
3. Install necessary package by conda
```
conda install ase matplotlib numpy pymatgen requests scipy beautifulsoup4 spglib prettytable jupyter
conda install pytorch pyg
```
4. Install e3nn package via [pip](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment)
```
pip install --upgrade-strategy only-if-needed e3nn
```
Finally, to install, without any dependency, use command: 
```bash
pip install --no-deps -e .
```
Tp uninstall, issue command:
```bash
pip uninstall automaticTB
```

### Installation of libmsym
Libmsym is used to discover molecular symmetry, to install, we can follow the exact procedure as 
given in the github project (https://github.com/mcodev31/libmsym) in our conda environment.
