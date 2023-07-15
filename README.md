# AutomaticTB

## Introduction

This package can find the non-equivalent interactions in a tight-binding model given the crystal structure. The tight-binding model can be solved to obtain band structure, density of states and be used for orbital interaction analysis.

## installation

1. Create an conda environment:
   
   ```bash
   conda create --name autoTB python=3.8
   conda activate autoTB
   ```

2. We need to add necessary channels
   
   ```bash
   conda config --add channels pyg pytorch conda-forge
   ```

3. Install necessary package by conda (torch is optional)
   
   ```bash
   conda install ase matplotlib numpy pymatgen requests scipy beautifulsoup4 spglib prettytable jupyter
   conda install pytorch pyg
   ```

4. Install e3nn package via [pip](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment)
   
   ```bash
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

   (e3nn package is used for rotation in vector space span by spherical harmonics)

## Others

Some part of the code is written in cython for performance. Before running the code, the cython module need to be compiled.

Go to `tools/cython` and compile by

```bash
python setup.py build_ext --inplace
```

