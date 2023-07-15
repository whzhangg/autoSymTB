
### methods

There are four core functionalities of this python module:

1. Character assignment for the symmetry operation in the site-symmetry group,
2. Formation of symmetrized linear combinations of atomic orbitals, and 
3. Handle symmetry relationship between different matrix elements in the Hamiltonian and overlap
4. Solution of electronic properties based on solving the Hamiltonian

### usage

The codebase serve mainly as a library, but core functionalities can be called using prepared scripts 

### code writting

Body scout rule: **leave your play ground better than you found it.**


### solving the band velocity

The band velocity is given by the dispersion relationship:

$$
v_n(k) = \frac{1}{\hbar}\nabla_k \varepsilon_n(k)
$$

which can be solved using finit difference method. If the unit of $\nabla_k \varepsilon_n(k)$ is eV$\cdot$A, the unit conversion factor would be `eV_in_SI * angstrom`.

The other method for solving the band velocity directly from the hamiltonian is given by equation 6) in Lee et al. 2018 *Tight-binding calculations of optical matrix element for conductivity using non-orthogonal atomic orbitals*. Equation 6) gives the expression of the band derivative.

### solving the transport properties

To find the transport property, we solve the Boltzmann equation following equation 5)--6) of the Boltzwannier paper *Pizzi et al. 2014*. Electrical conductivity, Seebeck coefficients and electronic thermal conductivity are calculated. Using a single band model, the calculation reproduce the result of the Weidemann and Franz Law, the seebeck coefficient value can also be reproduced at a level of $\approx 1\times10^2 \mu V/K$. 

Carrier concentration is calculated by summing over all the sampled kpoints with Fermi Draic distribution function at a given temperature

### interface design

The following electronic properties can be calculated: 

1. calculation of Hamiltonian and overlap matrix at arbitrary $k$ points (need speed up)
2. band structure
3. density of states (need speed up)
4. transport properties ($S$, $\sigma$, $\kappa_e$)
5. band effective mass around a given kpoints
6. DOS effective mass and mobility effective mass as given by Gibbs et al. 2017.

Currently, the class `ElectronicModel` is the interface that support the extraction of various properties.   It can be made to support the following methods:

- `ElectronicModel.Hamiltionian_at_k(k) -> [Hij(k),Sij(k)]`
- `ElectronicModel.plot_band_structure(kpath, emin, emax, shift) -> None`
- `ElectronicModel.get_dos(emin, emax, shift, nstep, grid_size) -> [x, y]`
- `ElectronicModel.band_effectivemass(k, ibnd) -> dict`
- `ElectronicModel.transport(grid_size, temps, mus, ncarriers, tau) -> TransportResult`

## Performance

### profiling

Profiling is performed to understand the execution bottleneck for the calculation of transport properties. Secular solver which compute the solution to the eigen equation is found to be the bottleneck. 

A further decomposition of `solver_secular` shows that the function that take most of the time is `fractional_matrix_power` used to find $S^{-1/2}$ (95% of the computation time). 

The `solver_secular` implementation is replaced by `eig` implemented in the `scipy.linalg` package which natively implements the generalized solution for the eigen equation.

As a result of profiling, `TightbindingModel` is re-written as `TightbindingModelOptimized` which a good speed improvement over the original implementation.

### multiprocessing

It is found that implementation of multiprocess (using concurrent) does not speed up implementation of the mathematic process in `solve_secular`. It seems that the implementation of numpy and scipy already utilize all the parallel capability of multi-core (vie multi-threading, as discussed [here](https://stackoverflow.com/questions/6941459/is-it-possible-to-know-which-scipy-numpy-functions-run-on-multiple-cores)) so that any additional parallalization will not speed up the process. Therefore, in the implementation of transport property calculation, multiprocessing is not included. 

## SISSO

The current code is integrated with the SISSO code through a provided interface. It aim to train and provide a parameterized description of the feature in band structure with a set of initial values. 

## Appendix

### installation of libmsym (not needed)

Libmsym is used to discover molecular symmetry, to install, we can follow the exact procedure as 
given in the github project ([GitHub - mcodev31/libmsym: molecular point group symmetry lib](https://github.com/mcodev31/libmsym)) in our conda environment.

It is necessary that we symmetrize the Hamiltonian `(h + np.conjugate(h.T)) / 2.0`, it's not yet added.

Bug:

the issue that cause the problem of Si in 2nn case is the identification of bond at the second neighbor. It should be noticed that for the secondary neighbor there is a rotational relationship of the local environment. 
