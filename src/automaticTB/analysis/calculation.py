"""
DONE: separate the execution to another class
DONE: set different delta value for E and V terms
DONE: use folder to contain results
DONE: write the code to analyze the results
DONE: do a line scan along L -> Sigma with tayler fitting
DONE: adjust the effective mass kgrid range
"""

import typing
import os
import dataclasses
import time

import numpy as np
import tqdm
import scipy

from automaticTB.interface import OrbitalPropertyRelationship
from automaticTB.properties import reciprocal
from automaticTB.properties import dos
from automaticTB.properties import bands
from automaticTB.properties import tightbinding
from automaticTB.properties import fermisurface
from automaticTB.properties import transport
from automaticTB.properties import calc_mesh
from automaticTB import tools


class BandAnalyzer:
    """this class generate data for training the regressor and sobol
    
    It take a `prefix.result` file as input as well as a set of 
    interaction values.
    """
    def __init__(self, 
        result_file: str,
        interactions: typing.Dict[str, float],
    ) -> None:
        self.prefix, _ = os.path.splitext(result_file)
        self.relationship = OrbitalPropertyRelationship.from_file(f"{self.prefix}.result")
        self.interactions = interactions
        self.initial_values = np.array([v for v in interactions.values()])


    @property
    def tb(self) -> tightbinding.TightBindingModel:
        """return the tightbinding parameter from initial values"""
        return self.relationship.get_tightbinding_from_free_parameters(
            free_Hijs=self.initial_values
        )


    def get_kmesh(self, nk: int, tetra: bool = False) -> reciprocal.Kmesh:
        """return kmesh or tetra mesh"""
        mesh = reciprocal.Kmesh.from_cell_nk(self.relationship.cell, nk)
        if tetra:
            mesh = dos.TetraKmesh(mesh.reciprocal_lattice, mesh.nks)
            
        return mesh


    def get_kpath(self, kpath_str: typing.List[str], band_quality: float) -> reciprocal.Kpath:
        """return kpath object"""
        return reciprocal.Kpath.from_cell_pathstring(
            self.relationship.cell, kpath_str, band_quality)


    def calculate_perturbed_bands(
        self, 
        perturbation: np.ndarray, 
        kpath: reciprocal.Kpath, 
        filename: str = ""
    ) -> bands.BandStructure:
        """calculate the band for the given kpath

        For each perturbation, the band is ordered, but we still need
        to check the ordering of individual bands. This can be done as
        a postprocessing step.

        Parameters
        ----------
        perturbations: array(nsample, nparams)
            zero centered perturbation to the initial parameters
        kpath: Kpath
            the path along which the energy is calculated
        write_original_band: bool
            whether to write the original band. 
            Filename will be `band_original.yaml`
        folder: str
            the folder to write all the band file

        Output
        ------
        None
            write each band structure `asdict(BandStructure)` to yaml file
        """
        tb = self.relationship.get_tightbinding_from_free_parameters(
            free_Hijs=perturbation + self.initial_values)
        
        banddata = bands.BandStructure.from_tightbinding_and_kpath(tb, kpath, order_band=True)
        if filename:
            tools.write_yaml(dataclasses.asdict(banddata), filename)
        return banddata


    def _get_suitable_kgrid(self, tb, kpoint, ibnd, de) -> typing.Tuple[np.ndarray, np.ndarray]:
        def _find_single_index(cs, c_ref) -> int:
            argmax = -1
            which = -1
            for ic, c in enumerate(cs.T):
                projection = np.abs(np.dot(c.conjugate(), c_ref))
                if projection > argmax:
                    argmax = projection
                    which = ic
            return which
        
        from automaticTB.properties.transport.band_effectivemass import create_small_delta_kmesh
        hijk, sijk, _, _ = tb.Hijk_Sijk_and_derivatives_at_k(kpoint)
        w, c = scipy.linalg.eig(hijk, sijk)
        sort_indices = np.argsort(w.real)
        e0 = w.real[sort_indices[ibnd]]
        reference_eigenvector = c[:,sort_indices[ibnd]]
        dk = 0.005
        for i in range(1, 100):
            dkgrid_frac, dkgrid_cart = create_small_delta_kmesh(
                self.relationship.cell, delta_frac=dk * i)
            kgrid = kpoint + dkgrid_frac
            hijks, sijks, _, _ = tb.Hijk_SijK_and_derivatives(kgrid)
            energies = np.zeros(len(kgrid))

            for ik in range(len(kgrid)):
                w, cs = scipy.linalg.eig(hijks[ik], sijks[ik])
                istate = _find_single_index(cs, reference_eigenvector)
                energies[ik] = w[istate].real
            
            average = np.mean(np.abs(energies - e0))
            if average > de and i > 1:
                return create_small_delta_kmesh(
                    self.relationship.cell, delta_frac=dk * (i-1))
        else:
            raise RuntimeError("the band is completely flat?")
            

    def _calc_effectivemass_taylor(
        self, kpoint: np.ndarray, ibnd: int, 
        parameters: np.ndarray, de: float
    ) -> np.ndarray:
        """
        return nsample of matrix 
        [e0, kx, ky, kz]
        [Vx, xx, xy, xz]
        [Vy, yx, yy, yz]
        [Vz, zx, zy, zz]
        """
        from scipy import optimize
        def _find_single_index(cs, c_ref) -> int:
            argmax = -1
            which = -1
            for ic, c in enumerate(cs.T):
                projection = np.abs(np.dot(c.conjugate(), c_ref))
                if projection > argmax:
                    argmax = projection
                    which = ic
            return which

        from automaticTB.properties.transport.band_effectivemass import fit_function_taylor
        
        rlv = reciprocal.find_RCL(self.relationship.cell)
        
        initial_model = self.relationship.get_ElectronicModel_from_free_parameters(
            free_Hijs=self.initial_values)
        hijk, sijk, _, _ = initial_model.tb.Hijk_Sijk_and_derivatives_at_k(kpoint)
        w, c = scipy.linalg.eig(hijk, sijk)
        sort_indices = np.argsort(w.real)
        reference_eigenvector = c[:,sort_indices[ibnd]]

        dkgrid_frac, dkgrid_cart = self._get_suitable_kgrid(initial_model.tb, kpoint, ibnd, de)

        nsobol = len(parameters)
        effectivemass44 = np.zeros((nsobol, 4, 4)) 
        # 0,0 is the energy, 1:3 is tensor
        for isample in tqdm.tqdm(range(nsobol)):
        #for isample in range(nsobol):
            model = self.relationship.get_ElectronicModel_from_free_parameters(
                free_Hijs=parameters[isample])

            k_start_cart = np.dot(rlv.T, kpoint)
            kgrid_frac = kpoint + dkgrid_frac

            hijk, sijk, hijk_derv, sijk_derv = model.tb.Hijk_Sijk_and_derivatives_at_k(kpoint)
            w, c = scipy.linalg.eig(hijk, sijk)
            ct = c.T
            derivative = np.zeros((len(w), 3))
            for i in range(len(w)):
                dhds = hijk_derv - w[i] * sijk_derv # (3, nbnd, nbnd)
                derivative[i,:] = np.einsum(
                    "i, kij, j -> k", np.conjugate(ct[i]), dhds, ct[i]).real
            
            istate = _find_single_index(c, reference_eigenvector)
            local_ref = c[:,istate]
            first_order = derivative[istate]
            k_energy = w[istate].real

            hijks, sijks, _, _ = model.tb.Hijk_SijK_and_derivatives(kgrid_frac)
            energies = np.zeros(len(kgrid_frac))
            for ik in range(len(kgrid_frac)):
                w, cs = scipy.linalg.eig(hijks[ik], sijks[ik])
                istate = _find_single_index(cs, local_ref)
                energies[ik] = w[istate].real

            dek = energies - k_energy \
                - first_order[0] * dkgrid_cart[:,0] \
                - first_order[1] * dkgrid_cart[:,1] \
                - first_order[2] * dkgrid_cart[:,2]
        
            guess = [0.0] * 6
            popt, _ = scipy.optimize.curve_fit(
                fit_function_taylor, dkgrid_cart, dek, p0=guess, maxfev=5000)
            dxx, dxy, dxz, dyy, dyz, dzz = popt

            effectivemass44[isample, 0, 0] = k_energy
            effectivemass44[isample, 0, 1:] = kpoint
            effectivemass44[isample, 1:, 0] = (first_order 
                * 1e-10 * scipy.constants.elementary_charge  / scipy.constants.hbar)
            
            effectivemass44[isample, 1:, 1:] = (
                np.array([[dxx, dxy, dxz], 
                          [dxy, dyy, dyz], 
                          [dxz, dyz, dzz]]) 
                    * scipy.constants.elementary_charge * (1e-10)**2 / scipy.constants.hbar**2
                    * scipy.constants.electron_mass)
        
        return effectivemass44


    def _calc_effectivemass_parabolic(
        self, kpoint: np.ndarray, ibnd: int, 
        parameters: np.ndarray, de: float
    ) -> np.ndarray:
        """
        return nsample of matrix 
        [e0, k0, k1, k2]
        [ 0, xx, xy, xz]
        [ 0, yx, yy, yz]
        [ 0, zx, zy, zz]
        """
        from scipy import optimize
        def _find_single_index(cs, c_ref) -> int:
            argmax = -1
            which = -1
            for ic, c in enumerate(cs.T):
                projection = np.abs(np.dot(c.conjugate(), c_ref))
                if projection > argmax:
                    argmax = projection
                    which = ic
            return which

        from automaticTB.properties.transport.band_effectivemass import fit_function
        
        rlv = reciprocal.find_RCL(self.relationship.cell)
        
        initial_model = self.relationship.get_ElectronicModel_from_free_parameters(
            free_Hijs=self.initial_values)

        hijk, sijk, _, _ = initial_model.tb.Hijk_Sijk_and_derivatives_at_k(kpoint)
        w, c = scipy.linalg.eig(hijk, sijk)
        sort_indices = np.argsort(w.real)
        reference_eigenvector = c[:,sort_indices[ibnd]]

        dkgrid_frac, dkgrid_cart = self._get_suitable_kgrid(initial_model.tb, kpoint, ibnd, de)

        nsobol = len(parameters)
        effectivemass44 = np.zeros((nsobol, 4, 4)) 
        # 0,0 is the energy, 1:3 is tensor
        for isample in tqdm.tqdm(range(nsobol)):
        #for isample in range(nsobol):
            model = self.relationship.get_ElectronicModel_from_free_parameters(
                free_Hijs=parameters[isample])

            k_start_frac = kpoint.copy()
            nstep = 2
            for _ in range(nstep):
                k_start_cart = np.dot(rlv.T, k_start_frac)
                kgrid_frac = k_start_frac + dkgrid_frac

                hijk, sijk, _, _ = model.tb.Hijk_Sijk_and_derivatives_at_k(kpoint)
                w, c = scipy.linalg.eig(hijk, sijk)
                istate = _find_single_index(c, reference_eigenvector)
                local_ref = c[:,istate]
                
                hijks, sijks, _, _ = model.tb.Hijk_SijK_and_derivatives(kgrid_frac)
                energies = np.zeros(len(kgrid_frac))

                for ik in range(len(kgrid_frac)):
                    w, cs = scipy.linalg.eig(hijks[ik], sijks[ik])
                    istate = _find_single_index(cs, local_ref)
                    energies[ik] = w[istate].real

                guess = [np.max(energies), *k_start_cart, 0, 0, 0, 0, 0, 0]
                popt, _ = optimize.curve_fit(
                    fit_function, k_start_cart + dkgrid_cart, energies, p0=guess, maxfev=5000)
                
                e0, k0x, k0y, k0z, dxx, dxy, dxz, dyy, dyz, dzz = popt
                kmin_frac = np.dot(np.linalg.inv(rlv.T), np.array([k0x, k0y, k0z]))
                inv_mass = (
                    np.array([[dxx, dxy, dxz], 
                              [dxy, dyy, dyz], 
                              [dxz, dyz, dzz]]) 
                    * scipy.constants.elementary_charge * (1e-10)**2 / scipy.constants.hbar**2
                    * scipy.constants.electron_mass
                )
                effectivemass44[isample,0,1:] = kmin_frac
                effectivemass44[isample,0,0] = e0
                effectivemass44[isample,1:,1:] = inv_mass
                k_start_frac = kmin_frac
        
        return effectivemass44


    def calculate_effective_mass(
        self, perturbations: np.ndarray, 
        kpoint: np.ndarray, 
        ibnd: int, 
        use_taylor: bool = True,
        de: float = 0.1
    ) -> None:
        """write effective mass around kpoints"""
        
        perturbed_values = self.initial_values + perturbations
        if use_taylor:
            effectivemass44 = self._calc_effectivemass_taylor(
                kpoint, ibnd, perturbed_values, de)
        else:
            effectivemass44 = self._calc_effectivemass_parabolic(
                kpoint, ibnd, perturbed_values, de)

        return effectivemass44


    def calculate_fermi_surface(
        self, 
        perturbation: np.ndarray, 
        kmesh: reciprocal.Kmesh, 
        filename: str = "", 
        use_ibz: bool = True
    ) -> fermisurface.FermiSurfaceData:
        """calculate fermi surface"""
        tb = self.relationship.get_tightbinding_from_free_parameters(
            free_Hijs=self.initial_values + perturbation
        )
        fs = fermisurface.FermiSurfaceData.from_tb_kmesh_singlespin(
            tb, kmesh, use_ibz=use_ibz
        )
        if filename:
            tools.write_yaml(dataclasses.asdict(fs), filename)
        
        return fs


    def calculate_perturbed_transport_crt(
        self,
        perturbation: np.ndarray,
        kmesh: reciprocal.Kmesh,
        nvb: int,
        temp: float,
        relative_mu: np.ndarray,
        filename: str = "",
        use_ibz: bool = True,
    ) -> dict:
        """calculate transport properties with constant tau"""

        tb = self.relationship.get_tightbinding_from_free_parameters(
            free_Hijs=self.initial_values + perturbation
        )
        t0 = time.time()
        if use_ibz:
            e, v = calc_mesh.calculate_e_v_using_ibz(tb, kmesh.kpoints, kmesh.nks)
        else:
            e, v, _ = tb.solveE_V_at_ks(kmesh.kpoints, average_degenerate=True)

        mu0 = np.max(e[:,0:nvb])
        mus = mu0 + relative_mu
        n_mus = len(mus)

        ncarrier = np.zeros(n_mus)
        sigma = np.zeros((n_mus, 3))
        seebeck = np.zeros((n_mus, 3))
        kappa = np.zeros((n_mus, 3))

        for i, mu in enumerate(mus):
            nc, sigma_ij, Sij, k_ij = transport.calculate_transport(
                mu=mu, temp=temp, nele=10, ek=e, vk=v, 
                tau=np.ones_like(e[:,:]), cell_volume=reciprocal.get_volume(tb.cell), weight=2.0
            )
            ncarrier[i] = nc / 1e6
            sigma[i] = np.diag(sigma_ij)
            seebeck[i] = np.diag(Sij)
            kappa[i] = np.diag(k_ij)

        time_taken = time.time() - t0
        data = {
            "mode": "crt",
            "perturbation": perturbation.tolist(),
            "nk": kmesh.nks,
            "temp": temp,
            "mus": mus.tolist(),
            "time": time_taken,
            "ncarrier": ncarrier.tolist(),
            "sigma": sigma.tolist(),
            "seebeck": seebeck.tolist(),
            "kappa": kappa.tolist()
        }
        if filename:
            tools.write_yaml(data, filename)
        return data


    def calculate_perturbed_transport_invg(
        self,
        perturbation: np.ndarray,
        kmesh: reciprocal.Kmesh,
        nvb: int,
        temp: float,
        relative_mu: np.ndarray,
        window_range: float = 2.0, 
        window_density: float = 250,
        min_scattering_rate: float = 1e-2,
        filename: str = "",
        use_ibz: bool = True,
    ) -> dict:
        """calculate transport properties with constant tau"""
        t0 = time.time()

        tb = self.relationship.get_tightbinding_from_free_parameters(
            free_Hijs=self.initial_values + perturbation
        )

        dosmesh = dos.TetraKmesh(kmesh.reciprocal_lattice, kmesh.nks)
        if use_ibz:
            e, v = calc_mesh.calculate_e_v_using_ibz(tb, dosmesh.kpoints, dosmesh.nks)
        else:
            e, v, _ = tb.solveE_V_at_ks(dosmesh.kpoints, average_degenerate=True)

        mu0 = np.max(e[:,0:nvb])
        doscalc = dos.TetraDOS(dosmesh, e, dos_weight=1.0) 
        # dos_weight is not important, set to 1
        dos_energy = np.linspace(
            mu0 - window_range,
            mu0 + window_range, 
            int(window_range * window_density * 2)
        )
        dos_result = doscalc.calculate_dos(dos_energy)
        dos_x = dos_result.x
        dos_y = dos_result.dos
        
        ph_scattering = np.zeros_like(e)
        nk, nbnd = e.shape
        for ik in range(nk):
            for ibnd in range(nbnd):
                if e[ik,ibnd] < mu0 + window_range and e[ik,ibnd] > mu0 - window_range:
                    min_index = np.argmin(np.abs(e[ik,ibnd] - dos_x))
                    ph_scattering[ik,ibnd] = max(min_scattering_rate, dos_y[min_index])
                else:
                    # outside the window range, scattering is set to be arbitrarily strong
                    ph_scattering[ik, ibnd] = 1e6
        tau = 1.0 / ph_scattering

        mus = mu0 + relative_mu
        n_mus = len(mus)

        ncarrier = np.zeros(n_mus)
        sigma = np.zeros((n_mus, 3))
        seebeck = np.zeros((n_mus, 3))
        kappa = np.zeros((n_mus, 3))

        for i, mu in enumerate(mus):
            nc, sigma_ij, Sij, k_ij = transport.calculate_transport(
                mu=mu, temp=temp, nele=10, ek=e, vk=v, 
                tau=tau, cell_volume=reciprocal.get_volume(tb.cell), weight=2.0
            )
            ncarrier[i] = nc / 1e6
            sigma[i] = np.diag(sigma_ij)
            seebeck[i] = np.diag(Sij)
            kappa[i] = np.diag(k_ij)

        time_taken = time.time() - t0
        data = {
            "perturbation": perturbation.tolist(),
            "nk": kmesh.nks,
            "temp": temp,
            "mus": mus.tolist(),
            "time": time_taken,
            "ncarrier": ncarrier.tolist(),
            "sigma": sigma.tolist(),
            "seebeck": seebeck.tolist(),
            "kappa": kappa.tolist(),
            "dosx": dos_x.tolist(),
            "dosy": dos_y.tolist()
        }
    
        if filename:
            tools.write_yaml(data, filename)
        return data