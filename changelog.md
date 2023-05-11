# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Since the project is still in development, changes are logged without version number

## 2023-04-09

### Changed

- In `analysis/calculation` array nk is changed to list nk.

## 2023-04-07

Major change to the properties subpackage.

### Added

- `TightBindingModel` now has `basis_name` and `nbasis` properties.
- weighted_mobility and effectivemass calculation is added to the transport calculation package. 

### Changed

- `OrbitalPropertyRelationship` now directly return a tightbinding object. 
- `TightBindingModel` is trimed to support only `solveE_at_ks` and `solveE_V_at_ks` method
- `TightBindingModel` now return the eigenvector as additional parameter.
- `TightBindingModel` now can average velocity at degenerate state as option (default is false).
- `BandStructure` result is now include band coefficient by default.
- `BandStructure`'s classmethod can order bands using eigenvectors
- kpoints.py is now renamed to reciprocal.py
- separate dos calculation (`TetraDOS`) and dos result (`DosResult`)
- properties module's `__init__.py` now does not include anything

### Removed

- Class object defined in `transport.py` is removed. use `calculate_transport()` instead
- `UnitCell` in kpoints.py (reciprocal.py) is now removed
- `ElectronicModel` class is now removed

## 2023-04-30

### Added

- added an experimental solver that should treat the conjugate relationship correctly.
- `LinearEquation` has a method now to return the solvable part of the homogeneous matrix.

## 2023-05-11

### Added

- `get_tightbinding_hijrs()` method in the interface class to obtain hijrs to constructure TB
- `degenerate_atomic_orbitals` tag in `solve()` to prevent enforcing degenerate on-site values
- corresponding changes are made to the `structure.py` file
- `plot_fatband()` now accept a enlargement factor for mark size
- a faster tightbinding `TightBindingModel_wo_overlap` to eliminate the need to create `SijR`s

### Changed

- remove the eigen-value checking in `solveE()`