import dataclasses
import typing

import numpy as np
import spglib

from automaticTB.properties import tightbinding
from automaticTB.properties import kpoints as kpt
from automaticTB import parameters
from automaticTB import tools

@dataclasses.dataclass
class MeshProperties:
    mapping: np.ndarray # nk,2: which k, which rotation
    kgrid: np.ndarray # (3,) int number of grid size
    kpoints: np.ndarray
    ibz_energies_velocity: np.ndarray # nk, nbnd, 4; first indice is energy
    cartesian_rotation: np.ndarray

    @classmethod
    def calculate_from_tb(
        cls, 
        tb: tightbinding.TightBindingModel, 
        nk: int
    ) -> "MeshProperties":
        cell = (tb.cell, tb.positions, tb.types)
        sym_data = spglib.get_symmetry_dataset(cell, symprec=parameters.spgtol)

        # point group operation
        unique = []
        for r in sym_data["rotations"]:
            found = False

            for r_u in unique:
                if np.allclose(r, r_u, atol=parameters.stol): 
                    found = True
                    break
            if found: continue
            unique.append(r)
        
        uc = kpt.UnitCell(tb.cell)
        c = tb.cell
        rc = uc.reciprocal_cell
        mesh = uc.recommend_kgrid(nk)

        nrot = len(unique)
        rot_k_frac = np.zeros((nrot, 3, 3))
        rot_k_cart = np.zeros((nrot, 3, 3))
        for ir, r in enumerate(unique):
            opc = c.T @ r @ np.linalg.inv(c.T)
            rot_frac_k = np.linalg.inv(rc.T) @ opc @ rc.T
            rot_k_cart[ir] = opc
            rot_k_frac[ir] = rot_frac_k
            #rot_frac_k = np.linalg.inv(rc.T).dot(c.T) @ r @ np.linalg.inv(c.T).dot(rc.T)
            #assert np.allclose(r, rot_frac_k), r - rot_frac_k
        
        mapping, grid = spglib.get_ir_reciprocal_mesh(
            mesh, cell, is_shift=[0, 0, 0], symprec=parameters.spgtol)

        grid = grid / mesh
        rotation_indices = tools.find_rotation_operation_indices(
            rot_k_frac, mapping, grid, parameters.stol
        )
        
        unique_kid = np.unique(mapping)
        calculated_properties = np.zeros((len(unique_kid),len(tb.basis),4))
        unique_kpoints = grid[unique_kid]
        e, v = tb.solveE_V_at_ks(unique_kpoints)
        calculated_properties[:,:,0] = e
        calculated_properties[:,:,1:] = v

        rev = {u: iu for iu, u in enumerate(unique_kid)}
        new_mapping = np.array([[rev[m],rotation_indices[im]] for im, m in enumerate(mapping)])
        #print(new_mapping)

        return cls(new_mapping, np.array(mesh), grid, calculated_properties, rot_k_cart)
    

    def get_unfolded_energies_velocities(self) -> np.ndarray:
        nk = len(self.mapping)
        _, nbnd, _ = self.ibz_energies_velocity.shape
        complete_grid = np.zeros((nk, nbnd, 4))
        for ik, (k_id, rot_id) in enumerate(self.mapping):
            rot = self.cartesian_rotation[rot_id]
            complete_grid[ik, :, 0] = self.ibz_energies_velocity[k_id, :, 0]
            for ibnd in range(nbnd):
                complete_grid[ik, ibnd, 1:] = rot @ self.ibz_energies_velocity[k_id, ibnd, 1:]
        return complete_grid
    

def straight_calculation(
        tb: tightbinding.TightBindingModel,  nk: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    cell = (tb.cell, tb.positions, tb.types)
    
    uc = kpt.UnitCell(tb.cell)
    mesh = uc.recommend_kgrid(nk)

    _, grid = spglib.get_ir_reciprocal_mesh(
        mesh, cell, is_shift=[0, 0, 0], symprec=parameters.spgtol)

    grid = grid / mesh
    nk = len(grid)
    nbnd = len(tb.basis)
    calculated_properties = np.zeros((nk, nbnd,4))
    e, v = tb.solveE_V_at_ks(grid)
    calculated_properties[:,:,0] = e
    calculated_properties[:,:,1:] = v

    return grid, calculated_properties
            
