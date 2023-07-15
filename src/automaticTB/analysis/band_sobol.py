import dataclasses
import typing

import tqdm
import numpy as np
from SALib.analyze import sobol

from automaticTB.properties import reciprocal
from ._analysis_params import which_sobol, calc_2order


@dataclasses.dataclass
class SobolBand:
    kpath: reciprocal.Kpath
    energies: np.ndarray
    problem: dict

    def calculate_sobol(
        self, 
        with_variance: bool = False, 
        bands_to_do: typing.Optional[typing.List[int]] = None
    ) -> np.ndarray:
        """return sobol indices [nk, nbnd, nfeature]"""
        
        _, nk, nbnd = self.energies.shape
        nfeature = self.problem["num_vars"]

        if not bands_to_do:
            bands_to_do = range(nbnd)

        nbnd = len(bands_to_do)

        sT_band = np.zeros((nk, nbnd, nfeature), dtype=float)

        for ik in tqdm.tqdm(range(nk)):
            for ibnd, band_index in enumerate(bands_to_do):
                res = sobol.analyze(
                    self.problem, self.energies[:,ik,band_index], calc_second_order=calc_2order)
                sT = res[which_sobol]
                factor = 1.0
                if with_variance:
                    factor = np.var(self.energies[:,ik, band_index])
                sT_band[ik, ibnd, :] = np.array(sT) * factor
        
        return sT_band
    

    def calculate_relative_sobol(
        self, 
        ref_k: np.ndarray, 
        ref_ibnd: int, 
        with_variance: bool = False,
        bands_to_do: typing.Optional[typing.List[int]] = None
    ) -> np.ndarray:
        """return relative sobol sensitivity relative to a k point"""
        
        kp = np.array(ref_k)
        dk = np.linalg.norm(self.kpath.kpoints - kp, axis=1)
        kid = np.argmin(dk)

        _, nk, nbnd = self.energies.shape
        nfeature = self.problem["num_vars"]

        if not bands_to_do:
            bands_to_do = range(nbnd)

        nbnd = len(bands_to_do)

        sT_band = np.zeros((nk, nbnd, nfeature), dtype=float)
        for ik in tqdm.tqdm(range(nk)):
            for ibnd, band_index in enumerate(bands_to_do):
                res = sobol.analyze(
                    self.problem, 
                    self.energies[:,ik,band_index] - self.energies[:,kid, ref_ibnd], 
                    calc_second_order=calc_2order
                )
                factor = 1.0
                if with_variance:
                    factor = np.var(self.energies[:,ik,band_index] - self.energies[:,kid, ref_ibnd])
                sT = res[which_sobol]
                sT_band[ik, ibnd, :] = np.array(sT) * factor
        return sT_band


    def get_sobol_for_kpoint_ibnd(self, kp: np.ndarray, ibnd: int) -> typing.Dict[str, float]:
        """get sobol for a single k point, return a dictionary"""
        kp = np.array(kp)
        dk = np.linalg.norm(self.kpath.kpoints - kp, axis=1)
        kid = np.argmin(dk)
        res = sobol.analyze(
            self.problem, self.energies[:, kid, ibnd], calc_second_order=calc_2order, seed =43)
        sort_index = np.argsort(-1 * res[which_sobol])
        return {self.problem["names"][i]: res[which_sobol][i] for i in sort_index}