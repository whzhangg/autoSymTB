import typing, dataclasses, collections
import numpy as np
from ..SALCs.linear_combination import Site, Orbitals, LinearCombination
from ..SALCs.vectorspace import VectorSpace
from ..structure.nncluster import CrystalSites_and_Equivalence
from ..structure.sites import CrystalSite


NamedLC = collections.namedtuple("NamedLC", "name lc")
BraKet = collections.namedtuple("BraKet", "bra ket")


@dataclasses.dataclass
class InteractionMatrix:
    states: list
    interactions: np.ndarray
    """
    this class help to find value in an interaction matrix, we store the data as a list of items
    and we should return the index by giving a pair item1, item2
    the item must have the __equal__ method so that we can use .index()
    """
    def get_index(self, state1, state2) -> typing.Tuple[int,int]:
        i = self.states.index(state1)
        j = self.states.index(state2)
        return (i, j)
    
    @classmethod
    def random_from_states(cls, states: list):
        nstate = len(states)
        coefficients = np.random.random((nstate, nstate))
        return cls(states, coefficients)

    @classmethod
    def zero_from_states(cls, states: list, dtype = float):
        nstate = len(states)
        coefficients = np.zeros((nstate, nstate), dtype=dtype)
        return cls(states, coefficients)

    @property
    def flattened_braket(self) -> typing.List[BraKet]:
        result = []
        for item1 in self.states:
            for item2 in self.states:
                result.append(
                    BraKet(item1, item2)
                )
        return result

    @property
    def flattened_interaction(self) -> np.ndarray:
        return self.interactions.flatten()


@dataclasses.dataclass
class MO:
    equivalent_index: int
    orbital: str
    irreps: str
    lc: LinearCombination

    @property
    def irrep_sequence(self) -> typing.List[str]:
        return self.irreps.split("->")

    def __eq__(self, o: object) -> bool:
        return self.equivalent_index == o.equivalent_index \
            and self.orbital == o.orbital \
            and self.irreps == o.irreps \
            and np.allclose(self.lc.coefficients, o.lc.coefficients)


@dataclasses.dataclass()
class AO:
    cluster_index: int
    primitive_index: int
    translation: np.ndarray
    chemical_symbol: int
    at_origin: bool
    l: int
    m: int

    def __eq__(self, o) -> bool:
        return self.primitive_index == o.primitive_index \
            and self.l == o.l and self.m == o.m \
            and np.allclose(self.translation, o.translation)


def get_AOlists_from_crystalsites_orbitals(csites: typing.List[CrystalSite], orbitals: typing.List[Orbitals]) \
-> typing.List[AO]:
    list_of_AOs = []
    num_sites = len(csites)
    zero = np.zeros(3, dtype=float)
    
    for i, csite, orb in zip(range(num_sites), csites, orbitals):
        at_origin = False
        if np.allclose(csite.site.pos, zero): at_origin = True
        for ao in orb.aolist:
            list_of_AOs.append(
                AO(i, csite.index_pcell, csite.translation, csite.site.chemical_symbol, at_origin, ao.l, ao.m)
            )
    return list_of_AOs

# this should be an intermediate method that transform MO and AO
class MOCoefficient:
    def __init__(self, 
        sitewithequivalence: CrystalSites_and_Equivalence, labelledvectorspace: typing.Dict[str, VectorSpace]
    ):
        list_named_lcs = _get_namedLC_from_decomposed_vectorspace(labelledvectorspace)
        orbitals: typing.List[Orbitals] = list_named_lcs[0].lc.orbitals
        coefficient_matrix = np.vstack(
            [ namedlc.lc.coefficients for namedlc in list_named_lcs ]
        )

        # make MOs
        equivalent_atoms = sitewithequivalence.equivalent_atoms
        named_subspace = _divide_subspace(equivalent_atoms, orbitals, coefficient_matrix)
        self._list_of_MOs: typing.List[MO] = []
        for namedlc, ao_name in zip(list_named_lcs, named_subspace):
            self._list_of_MOs.append(
                MO(ao_name[0], ao_name[1], namedlc.name, namedlc.lc)
            )

        # make AOs
        self._list_of_AOs = get_AOlists_from_crystalsites_orbitals(sitewithequivalence.crystalsites, orbitals)

        
        # make matrix A
        matrix_A = []
        for _, lcc_i in enumerate(coefficient_matrix):
            for _, lcc_j in enumerate(coefficient_matrix):
                matrix_A.append(
                    np.tensordot(lcc_i, lcc_j, axes=0).flatten()
                )
        self._matrix_A = np.vstack(matrix_A)
        assert self._matrix_A.shape[0] == self._matrix_A.shape[1]

    def save(self, fn: str):
        # AOs can be saved conveniently, but MOs are difficult to save
        pass

    @classmethod
    def from_file(self, fn: str): # -> MOCoefficient
        pass


    @property
    def AOs(self) -> typing.List[AO]:
        return self._list_of_AOs

    @property
    def MOs(self) -> typing.List[MO]:
        return self._list_of_MOs

    @property
    def matrixA(self) -> np.ndarray:
        return self._matrix_A

    @property
    def inv_matrixA(self) -> np.ndarray:
        return np.linalg.inv(self._matrix_A)


def AO_from_MO(coefficient: MOCoefficient, MOinteraction: InteractionMatrix) -> InteractionMatrix:
    aos = coefficient.AOs
    h_ij = np.dot(coefficient.inv_matrixA, MOinteraction.flattened_interaction)
    h_ij = h_ij.reshape((len(aos), len(aos)))
    return InteractionMatrix(aos, h_ij)


def MO_from_AO(coefficient: MOCoefficient, AOinteraction: InteractionMatrix) -> InteractionMatrix:
    mos = coefficient.MOs
    h_ij = np.dot(coefficient.matrixA, AOinteraction.flattened_interaction)
    h_ij = h_ij.reshape((len(mos), len(mos)))
    return InteractionMatrix(mos, h_ij)


def _get_namedLC_from_decomposed_vectorspace(vc_dicts: typing.Dict[str, VectorSpace]) -> typing.List[NamedLC]:
    result = []
    for irrepname, vs in vc_dicts.items():
        for lc, i in zip(vs.get_nonzero_linear_combinations(), range(vs.rank)):
            # only append the rank, temperory solution?
            result.append(
                NamedLC(irrepname, lc.get_normalized())
            )
    return result


def _divide_subspace(equivalent_dict: typing.Dict[int, typing.Set[int]], 
                    orbitals: typing.List[Orbitals], coefficients: np.ndarray) \
    -> typing.List[tuple]: # list of (i, "s"), i is the index of equivalent atom in the sites
    start = 0
    range_dicts = []
    
    for i, orb in enumerate(orbitals):
        subslice = orb.slice_dict
        range_dict = {}
        for k, s in subslice.items():
            end = start + s.stop - s.start
            range_dict[k] = list(range(start, end))
            start = end
        range_dicts.append(range_dict)

    eq_orb_slice = {}
    for ieq, eqset in equivalent_dict.items():
        orb = orbitals[ieq]
        for o in orb.orblist:
            all_index = []
            for eq_atom in eqset:
                all_index += range_dicts[eq_atom][o]
            eq_orb_slice[(ieq, o)] = all_index

    row_name = []
    subspace_namelist = list(eq_orb_slice.keys())
    for row in coefficients:
        row_correspond: typing.List[bool] = []
        for _, indices in eq_orb_slice.items():
            row_correspond.append(np.linalg.norm(row[indices]) > 1e-6)
        assert row_correspond.count(True) == 1
        row_name.append(
            subspace_namelist[row_correspond.index(True)]
        )
    
    return row_name

