import typing, abc, numpy as np

from automaticTB.tools import LinearEquation, tensor_dot
from automaticTB.parameters import zero_tolerance, complex_coefficient_type
from automaticTB.solve.SALCs import IrrepSymbol
from .interaction_pairs import AOPair, AOPairWithValue, AOSubspace


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
                self.all_AOpairs[i] for i in self.homogeneous_equation.free_variable_indices
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
            all_solved_values = self.homogeneous_equation.solve_with_values(values)
            
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



class SimpleInteraction(InteractionBase):
    def __init__(self, allpairs: typing.List[AOPair], matrix: np.ndarray) -> None:
        self._allpairs = allpairs
        self._linear = LinearEquation.from_equation(matrix)

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._linear


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._allpairs
    

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
            linear_equation = LinearEquation.from_equation(homogeneous_part)

        return linear_equation, non_homogeneous

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation

    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs


class BlockInteractions(InteractionBase):
    """This class combineds the block diagonal form"""
    def __init__(self, interactions: typing.List[InteractionBase]) -> None:
        self._all_aopairs = []

        self._block_diagonal_size = []
        for interaction in interactions:
            self._all_aopairs += interaction.all_AOpairs
            homo_nrow = 0
            if interaction.homogeneous_equation is not None:
                homo_nrow = len(interaction.homogeneous_matrix)
            self._block_diagonal_size.append(
                (
                    len(interaction.all_AOpairs), 
                    homo_nrow
                )
            )

        nao = len(self._all_aopairs)
        nrow_homo = sum(n for _,n in self._block_diagonal_size)

        homo_matrix = np.zeros((nrow_homo, nao), dtype=complex_coefficient_type)

        nr1_start = 0 # homogeneous matrix
        col_start = 0
        for (ncol, nr1), interaction in zip(self._block_diagonal_size, interactions):
                
            nr1_end = nr1_start + nr1
            col_end = col_start + ncol

            if nr1 > 0:
                homo_matrix[nr1_start:nr1_end, col_start:col_end] \
                    = interaction.homogeneous_matrix

            col_start = col_end
            nr1_start = nr1_end
            
        self._homogeneous_equation = LinearEquation.from_equation(homo_matrix)

    @property
    def homogeneous_equation(self) -> LinearEquation:
        return self._homogeneous_equation


    @property
    def all_AOpairs(self) -> typing.List[AOPair]:
        return self._all_aopairs


    def print_log(self) -> None:
        print( '## Block Diagonal Interaction matrix')
        print( '  Homogeneous matrix is direct sum of:')
        tmp_out = []
        for ncol, nr1 in self._block_diagonal_size:
            tmp_out.append(f"({nr1:3>d} x {ncol:3>d})")
        print('  ' + " + ".join(tmp_out))
        print("")
        super().print_log()