import typing, abc, dataclasses, copy
import numpy as np

from automaticTB.tools import LinearEquation, tensor_dot
from automaticTB.parameters import zero_tolerance
from automaticTB.solve.SALCs import IrrepSymbol
from automaticTB.solve.structure import CenteredCluster
from .interaction_pairs import AO, AOPair, AOPairWithValue, AOSubspace


__all__ = ["InteractionBase", "InteractingAOSubspace", "InteractingAOSubspaceTranslation"]


class InteractionBase(abc.ABC):
    """Base class of interaction guarantee properties and some methods

    Defines a InteractionSubspace interface that guarantees the `.homogeneous_equation`, `.non_homogeneous` and the `.all_AOpairs` variable. It also provide the method to obtain the solution, so that we do not need to redefine the same method afterwards. 
    """
    
    @property
    @abc.abstractmethod
    def homogeneous_equation(self) -> typing.Optional[LinearEquation]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def all_AOpairs(self) -> typing.List[AOPair]:
        raise NotImplementedError()

    @property
    def homogeneous_matrix(self) -> np.ndarray:
        return self.homogeneous_equation.row_echelon_form

    @property
    def free_AOpairs(self) -> typing.List[AOPair]:
        if self.homogeneous_equation is None:
            # no relationship, all pairs are dependent
            return self.all_AOpairs
        else:
            # indices are sorted by LinearEquation
            return [
                self.all_AOpairs[i] for i in self.homogeneous_equation.free_variables_index
            ]


    def print_all_AOpairs(self) -> None:
        for i, f in enumerate(self.all_AOpairs):
            print(f"{i+1:>3d} " + str(f))


    def print_free_AOpairs(self) -> None:
        for i, f in enumerate(self.free_AOpairs):
            print(f"{i+1:>3d} " + str(f))


    def print_log(self) -> None:
        print( "## Atomic Orbital Interactions")
        print(f"  (free/total) interactions: {len(self.free_AOpairs)}/{len(self.all_AOpairs)}")
        if self.free_AOpairs:
            for i, f in enumerate(self.free_AOpairs):
                print(f"  {i+1:>3d} " + str(f))
        else:
            raise RuntimeError("free interaction are not obtained? (InteractionBase)")


    def solve_interactions_to_InteractionPairs(self, values: typing.List[float]) \
    -> typing.List[AOPairWithValue]:
        """solve all interaction by calling the homogeneous_equation"""
        if self.homogeneous_equation is None:
            all_solved_values = values
        else:
            values = np.array(values)
            all_solved_values = self.homogeneous_equation.solve_providing_values(values)
            
        if len(all_solved_values) == len(self.all_AOpairs):
            raise RuntimeError(
                "(InteractionBase.solve_interactions) " 
                + "number of solved value are different from the total number of values"
            )
        
        aopairs_withvalue = []
        for pair, value in zip(self.all_AOpairs, all_solved_values):
            aopairs_withvalue.append(
                AOPairWithValue(pair.l_AO, pair.r_AO, value)
            )
        return aopairs_withvalue


    def print_debug_info(self,
        print_free: bool = False,
        print_all: bool = False,
    ) -> None:
        print("*" * 60)
        print(f"total number of AO pairs needed: {len(self.all_AOpairs)}")
        print("")
        homo_size = self.homogeneous_part.shape
        homo_rank = np.linalg.matrix_rank(
            self.homogeneous_equation.homogeneous_equation, tol = zero_tolerance)
        print(f"Homogeneous part size: ({homo_size[0]},{homo_size[1]})")
        print(f"                 rank: {homo_rank}")
        print("")
        if print_free:
            print(f"Number of free parameter: {len(self.free_AOpairs)}")
            for i, pair in enumerate(self.free_AOpairs):
                print(f"{i+1:>2d} " + str(pair))
        print("*" * 60)
        if print_all:
            print(f"The considered pairs: ")
            for i, pair in enumerate(self.all_AOpairs):
                print(f"{i+1:>2d} " + str(pair))


class InteractingAOSubspace(InteractionBase):
    _forbidden_symbol = 0

    def __init__(self, l_subspace: AOSubspace, r_subspace: AOSubspace) -> None:
        self.l_subspace = l_subspace
        self.r_subspace = r_subspace
        
        self._all_aopairs = [ 
            AOPair.from_pair(p) for p in tensor_dot(self.l_subspace.aos, self.r_subspace.aos) 
        ]
        
        reps_tp_list = []
        for l_nlc in self.l_subspace.namedlcs:
            for r_nlc in self.r_subspace.namedlcs:
                l_rep = l_nlc.name
                r_rep = r_nlc.name
                tp = np.tensordot(
                    l_nlc.lc.coefficients, 
                    r_nlc.lc.coefficients,
                    axes=0
                ).flatten()
                reps_tp_list.append(
                    (l_rep, r_rep, tp)
                )

        self._homogeneous_equation, self._nonhomogeneous = \
            self.obtain_homogeneous_nonhomogeneous_equation(reps_tp_list)
        
    def obtain_homogeneous_nonhomogeneous_equation(self, 
        reps_tp_list: typing.List[typing.Tuple[IrrepSymbol, IrrepSymbol, np.ndarray]]
    ) -> typing.Tuple[LinearEquation, np.ndarray]:
        """obtain the homogeneous relationship as a LinearEquation and the inhomogeneous matrix"""
        references: typing.List[int] = []
        tps = np.array([tp for _, _, tp in reps_tp_list])
        memory = {}
        symbols = 1
        for rep1, rep2, _ in reps_tp_list:
            if rep1.symmetry_symbol != rep2.symmetry_symbol:
                references.append(self._forbidden_symbol)
            else:
                main1 = f"{rep1.main_irrep}^{rep1.main_index}"
                main2 = f"{rep2.main_irrep}^{rep2.main_index}"
                pair = " ".join([main1, main2])
                if pair not in memory:
                    memory[pair] = symbols
                    references.append(symbols)
                    symbols += 1
                else:
                    references.append(memory[pair])
        
        non_homogeneous  = []
        homogeneous_part = [
            tps[i] for i, type in enumerate(references) if type == self._forbidden_symbol
        ]
        all_types = set(references) - set([self._forbidden_symbol])
        for thetype in all_types:
            indices = [i for i, type in enumerate(references) if type == thetype]
            if len(indices) > 1:
                for i in indices[1:]:
                    homogeneous_part.append(tps[i] - tps[indices[0]])
            non_homogeneous.append(tps[indices[0]])

        if len(homogeneous_part) + len(non_homogeneous) != len(tps):
            raise RuntimeError("_obtain_homogeneous_nonhomogeneous_equation " + \
                "separation of homo-inhomo parts is not successful")
        
        homogeneous_part = np.array(homogeneous_part)
        non_homogeneous = np.array(non_homogeneous)

        linear_equation = None
        if homogeneous_part.size > 0:
            linear_equation = LinearEquation(homogeneous_part)

        return linear_equation, non_homogeneous

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs


class InteractingAOSubspaceTranslation(InteractionBase):
    def __init__(self, solved_subspace: InteractingAOSubspace) -> None:

        self._all_aopairs = solved_subspace.all_AOpairs
        if solved_subspace.homogeneous_equation is None:
            self._homogeneous_equation = solved_subspace.homogeneous_equation
            return 
        know_homogeneous = solved_subspace.homogeneous_matrix

        all_sites = set()
        for aopair in self._all_aopairs:
            all_sites.add(Site.from_AO(aopair.l_AO))
            all_sites.add(Site.from_AO(aopair.r_AO))
        all_sites: typing.List[Site] = list(all_sites)

        rows = []
        pair_index = {aopair: ipair for ipair, aopair in enumerate(self._all_aopairs)}
        print("-----")
        for ipair, pair in enumerate(self._all_aopairs):
            print(f"{ipair+1:>2d}" + str(pair))
        for ipair, aopair in enumerate(self._all_aopairs):
            l_ao = aopair.l_AO # inside unit cell
            r_ao = aopair.r_AO # 

            if r_ao.primitive_index == l_ao.primitive_index \
                and np.linalg.norm(r_ao.translation) > zero_tolerance:
                print(aopair)
                reverse_pair = get_translated_AO(AOPair(r_ao, l_ao))
                print(reverse_pair)
                index = pair_index[reverse_pair]
                new_row = np.zeros(len(self._all_aopairs), dtype=know_homogeneous.dtype)
                new_row[ipair] += 1.0
                new_row[index] -= 1.0
                rows.append(new_row)
        
        if len(rows) > 0:
            self._homogeneous_equation = LinearEquation(
                np.vstack([know_homogeneous, np.array(rows)])
            )
        else:
            self._homogeneous_equation = LinearEquation(know_homogeneous)
    
    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs

    @property
    def homogeneous_equation(self) -> typing.Optional[LinearEquation]:
        return self._homogeneous_equation

@dataclasses.dataclass(eq=True)
class Site:
    pindex: int
    eqindex: int
    t1: int 
    t2: int
    t3: int
    abs_pos: np.ndarray
    symbol: str

    @classmethod
    def from_AO(cls, ao: AO) -> "Site":
        return cls(
            pindex = ao.primitive_index,
            eqindex = ao.equivalent_index,
            t1 = int(ao.translation[0]),
            t2 = int(ao.translation[1]),
            t3 = int(ao.translation[2]),
            abs_pos = ao.absolute_position,
            symbol = ao.chemical_symbol
        )

    @classmethod
    def from_cluster(cls, cluster: CenteredCluster) -> "Site":
        center = cluster.center_site
        return cls(
            pindex = center.index_pcell,
            eqindex = center.equivalent_index,
            t1 = int(center.translation[0]),
            t2 = int(center.translation[1]),
            t3 = int(center.translation[2]),
            abs_pos = center.absolute_position,
            symbol = center.site.chemical_symbol
        )

    def as_tuple(self) -> tuple:
        return (self.pindex, self.t1, self.t2, self.t3)

    def __eq__(self, o: "Site") -> bool:
        return self.as_tuple() == o.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())


def get_translated_AO(aopair: AOPair) -> AOPair:
    l_ao = copy.deepcopy(aopair.l_AO)
    r_ao = copy.deepcopy(aopair.r_AO)
    r_ao.translation -= l_ao.translation
    l_ao.translation = np.zeros(3)
    return AOPair(l_ao, r_ao)
