import numpy as np
import os, typing

from automaticTB.structure.structure import Structure
from automaticTB.SALCs.vectorspace import get_vectorspace_from_NNCluster
from automaticTB.SALCs.decompose import decompose_vectorspace
from automaticTB.tightbinding.MOcoefficient import (
    MOCoefficient, get_AOlists_from_crystalsites_orbitals, InteractionMatrix,
    AO_from_MO, MO_from_AO
)
from automaticTB.tightbinding.tightbinding_model import make_HijR_list, TightBindingModel
from automaticTB.utilities import print_matrix, find_RCL

from structure import cell, types, positions, orbits_used
from ao_interaction import obtain_AO_interaction_from_AOlists

structure: Structure = Structure.from_cpt_rcut(cell, positions, types, 3.0)  
reciprocal_cell = find_RCL(structure.cell)
# distance Pb - Cl is ~2.8 A


def get_Perovskite_MOcoefficients() \
-> typing.List[MOCoefficient]:
    from automaticTB.utilities import save_yaml, load_yaml
    filename = "Perovskite_MOcoeff.yml"

    if os.path.exists(filename):
        return load_yaml(filename)
    
    coefficients = []

    for cluster in structure.nnclusters:
        if len(cluster.crystalsites) == 7: continue
        print(cluster.equivalent_atoms_reference)
        print(cluster.sitesymmetrygroup.groupname)
        print(cluster.orbitalslist)
        raise
        group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
        site_with_equivalence = cluster.get_CrystalSites_and_Equivalence_from_group(group)
        vectorspace = get_vectorspace_from_NNCluster(cluster, orbits_used)
        decomposed = decompose_vectorspace(vectorspace, group)

        mocoeff = MOCoefficient(site_with_equivalence, decomposed)
        coefficients.append(mocoeff)

    save_yaml(coefficients, filename)
    return coefficients
    


def get_Perovskite_tight_binding_model_from_AO_interaction():
    c, p, t = structure.cpt

    interactions = []

    for cluster in structure.nnclusters:
        assert len(cluster.crystalsites) in [3, 7]
        group = cluster.get_SiteSymmetryGroup_from_possible_rotations(structure.cartesian_rotations)
        assert group.groupname in ["m-3m", "4/mmm"]
        
        vectorspace = get_vectorspace_from_NNCluster(cluster, orbits_used)
        orbitals = vectorspace.orbitals

        aos = get_AOlists_from_crystalsites_orbitals(cluster.crystalsites, orbitals)
        interaction = obtain_AO_interaction_from_AOlists(c, p, aos)
        interactions.append(interaction)
        assert np.allclose(interaction.interactions, interaction.interactions.T)

    hijr_list = make_HijR_list(interactions)

    model = TightBindingModel(c, p, t, hijr_list)
    return model 


def get_Perovskite_tight_binding_model_from_MO_interaction():
    c, p, t = structure.cpt
    MOcoefficients = get_Perovskite_MOcoefficients()
    for mocoeff in MOcoefficients:
        ao_interaction = obtain_AO_interaction_from_AOlists(c, p, mocoeff.AOs)
        mo_interaction = MO_from_AO(mocoeff, ao_interaction)
        # check that AO_from_MO is indeed the inverse of MO_from_AO
        #ao_re = AO_from_MO(mocoeff, mo_interaction)
        #assert np.allclose(ao_re.interactions, ao_interaction.interactions)
        print_matrix(ao_interaction.interactions, "{:>6.2f}")
        for pair, value in zip(mo_interaction.flattened_pair, mo_interaction.flattened_interaction):
            if abs(value) > 1e-6:
                print(pair.left.equivalent_index, pair.left.orbital, pair.left.at_origin)
                print(pair.right.equivalent_index, pair.right.orbital, pair.right.at_origin)
                print("---->", value)

        raise


def obtain_and_plot_bandstructure(model: TightBindingModel, filename: str = "HalidePerovskite_band.pdf") -> None:
    from automaticTB.properties.bandstructure import Kline, Kpath, get_bandstructure_result

    # as in the paper of 
    # Symmetry-Based Tight Binding Modeling of Halide Perovskite Semiconductors
    kpos = {
        "M": np.array([0.5,0.5,0.0]),
        "X": np.array([0.5,0.0,0.0]),
        "G": np.array([0.0,0.0,0.0]),
        "R": np.array([0.5,0.5,0.5]),
    }

    lines = [
        Kline("M", kpos["M"], "R", kpos["R"], 10),
        Kline("R", kpos["R"], "G", kpos["G"], 18),
        Kline("G", kpos["G"], "X", kpos["X"], 10),
        Kline("X", kpos["X"], "M", kpos["M"], 10),
        Kline("M", kpos["M"], "G", kpos["G"], 15),
    ]

    kpath = Kpath(reciprocal_cell , lines)

    band_result = get_bandstructure_result(model, kpath)
    band_result.plot_data(filename)


def obtain_and_plot_dos(model: TightBindingModel, filename: str = "HalidePerovskite_dos.pdf") -> None:
    from automaticTB.properties.dos import get_tetrados_result
    tetrados = get_tetrados_result(model, ngrid=np.array([10,10,10]))
    tetrados.plot_data(filename)


if __name__ == "__main__":
    get_Perovskite_MOcoefficients()