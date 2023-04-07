import numpy as np, typing, dataclasses, copy

from automaticTB import parameters as params
from automaticTB import tools

@dataclasses.dataclass(eq=True)
class AtomicOrbital:
    pindex: int
    n: int 
    l: int
    m: int
    t1: int
    t2: int
    t3: int

    def as_tuple(self):
        return (self.pindex, self.n, self.l, self.m, self.t1, self.t2, self.t3)

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.t1, self.t2, self.t3])

    def as_line(self) -> str:
        return "{:>3d}{:>3d}{:>3d}{:>3d} t = {:>3d}{:>3d}{:>3d}".format(*self.as_tuple())

    @classmethod
    def from_line(cls, aline) -> "AtomicOrbital":
        indices, translation = aline.strip().split("t =") 
        idx = [ int(i) for i in indices.split() ]
        trans = [ int(t) for t in translation.split() ]
        return cls(*idx, *trans)

    def __repr__(self) -> str:
        return "{:>4d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d}".format(*self.as_tuple())

    @classmethod
    def from_str(cls, string: str) -> "AtomicOrbital":
        parts = [int(p) for p in string.split()]
        return cls(*parts)


@dataclasses.dataclass
class OrbitalPair:
    l_orbital: AtomicOrbital
    r_orbital: AtomicOrbital

    def get_reverse_translated(self) -> "OrbitalPair":
        l_orbital = copy.deepcopy(self.l_orbital)
        r_orbital = copy.deepcopy(self.r_orbital)
        l_orbital.t1 -= r_orbital.t1
        l_orbital.t2 -= r_orbital.t2
        l_orbital.t3 -= r_orbital.t3
        r_orbital.t1 = 0; r_orbital.t2 = 0; r_orbital.t3 = 0
        return OrbitalPair(r_orbital, l_orbital)

    def __hash__(self) -> int:
        return hash(self.l_orbital.as_tuple() + self.r_orbital.as_tuple())

    def __eq__(self, o: "OrbitalPair") -> bool:
        return self.l_orbital == o.l_orbital and self.r_orbital == o.r_orbital

    def __repr__(self) -> str:
        return str(self.l_orbital) + " -> " + str(self.r_orbital)

    @classmethod
    def from_str(cls, aline) -> "OrbitalPair":
        left, right = aline.split("->")
        return cls(
            AtomicOrbital.from_str(left), 
            AtomicOrbital.from_str(right)
        )


@dataclasses.dataclass
class SolutionChain:
    """store a direct numerical relationship for solving interactions"""
    inv_m1: np.ndarray # square matrix
    inv_m2: typing.Optional[np.ndarray]
    extra_value_finder: typing.Optional[typing.Dict[int, bool]] # indices, 

    def _create_b_vector(self, 
            totalsize: int, list_of_values: typing.List[np.ndarray]) -> np.ndarray:
        """append list of values to the end"""
        b = np.zeros(totalsize, dtype = self.inv_m1.dtype)
        end = totalsize
        for values in list_of_values[::-1]:
            start = end - len(values)
            b[start:end] = values
            end = start
        return b

    def solve(self, input_values: typing.List[float]) -> np.ndarray:
        """go through the chain of action"""
        b = self._create_b_vector(len(self.inv_m1), [np.array(input_values)])

        x0 = self.inv_m1 @ b
        if self.inv_m2 is None:
            return x0
        
        additional_values = []
        for i, do_conjugate in self.extra_value_finder.items():
            _v = x0[i]
            if do_conjugate:
                _v = np.conjugate(_v)
            additional_values.append(_v)
        
        fullb = self._create_b_vector(
            len(self.inv_m2), [np.array(input_values), np.array(additional_values)])
        
        return self.inv_m2 @ fullb


#@dataclasses.dataclass
#class OrbitalPropertyRelationship:
    """
    it can be converted back and forth with the interactionSpace class, 
    it support saving to file and reading from file, it calles the 
    InteractionSpace to do real work and and provide the method 
    `get_ElectronicModel_from_free_parameters()` which interface 
    with the property module.
    """
    #cell: np.ndarray
    #positions: np.ndarray
    #types: np.ndarray
    #all_pairs: typing.List[OrbitalPair]
    #free_pair_indices: typing.List[int]
    #homogeneous_equation: np.ndarray


class OrbitalPropertyRelationship:
    def __init__(self,
        cell: np.ndarray,
        positions: np.ndarray,
        types: np.ndarray,
        all_pairs: typing.List[OrbitalPair],
        free_pair_indices: typing.List[int],
        homogeneous_equation: np.ndarray,
    ) -> None:
        self.cell = cell
        self.positions = positions
        self.types = types
        self.all_pairs = all_pairs
        self.free_pair_indices = free_pair_indices
        self.homogeneous_equation = homogeneous_equation

        self._stored_solutionchain = None

    def _extend_matrix(self, a: np.ndarray, indices: typing.List[int]) -> np.ndarray:
        """extend matrix by index
        
        a: [1, 0, 1] and indices: [1, 0], will return
            [1, 0, 1]
            [0 ,1 ,0]
            [1, 0, 0]
        """
        _, ncol = a.shape
        additional = np.zeros((len(indices), ncol), dtype=a.dtype)
        for irow, icol in enumerate(indices):
            additional[irow, icol] = 1.0
        return np.vstack([a, additional])
        

    def _create_solutionchain(self) -> SolutionChain:
        # obtain additional_required_indices if any
        tmp = tools.LinearEquation.from_equation(
            self._extend_matrix(self.homogeneous_equation, self.free_pair_indices)
        )
        
        additional_required_indices = tmp.free_variable_indices

        if len(additional_required_indices) == 0:
            solvable_inverse = np.linalg.inv(tmp.row_echelon_form)
            return SolutionChain(solvable_inverse, None, None)
        
        # find the solvable matrix 
        related_indices = set()
        indices = np.array(additional_required_indices)
        for row in self.homogeneous_equation:
            if np.all(np.isclose(row[indices], 0.0, atol = params.ztol)):
                # a row does not containing any of the no-solvable indices
                continue
            related_indices |= set(
                np.argwhere(np.invert(np.isclose(row, 0.0, atol=params.ztol))).flatten()
            )
            
            # AOs that depend on the selected additional_required_index

        solvable_indices = sorted(list(set(range(len(self.all_pairs))) - related_indices))
        for i in self.free_pair_indices:
            if i not in solvable_indices:
                raise ValueError("_create_solutionchain, solvable part is probably not correct!")

        map_from_old_indices = {old: new for new, old in enumerate(solvable_indices)}

        solvable_parts = tools.LinearEquation.from_equation(
            self.homogeneous_equation[:, np.array(solvable_indices)]
        )
        free_indices_for_solvable_part = [
            map_from_old_indices[i] for i in self.free_pair_indices]

        stacked_left = self._extend_matrix(
            solvable_parts.row_echelon_form, free_indices_for_solvable_part)
        solvable_inverse = np.linalg.inv(stacked_left)

        solved_aopair_values = {self.all_pairs[si]:i for i, si in enumerate(solvable_indices)} 

        additional_values_index: typing.Dict[int, bool] = dict()
        for additional_rquired_i in additional_required_indices:
            pair = self.all_pairs[additional_rquired_i]
            reverse_pair = pair.get_reverse_translated()
            if pair in solved_aopair_values:
                additional_values_index[solved_aopair_values[pair]] = False
            elif reverse_pair in solved_aopair_values:
                additional_values_index[solved_aopair_values[reverse_pair]] = True
            else:
                raise RuntimeError("interaction pair not found")
        
        stacked_left = self._extend_matrix(
            self.homogeneous_equation, self.free_pair_indices + additional_required_indices
        )
        
        full_inverse = np.linalg.inv(stacked_left)

        return SolutionChain(
                solvable_inverse, 
                full_inverse,
                additional_values_index,
            )


    def _solve_all_values(self, values: typing.List[float]) -> np.ndarray:
        all_indices = range(len(self.all_pairs))
        dtype = self.homogeneous_equation.dtype

        if len(values) != len(self.free_pair_indices):
            raise RuntimeError(
                "{:s}: number of input values of interaction {:d} != required {:d}".format(
                    self.__name__, len(values), len(self.free_pair_indices)
                )
            )
    
        if not self.free_pair_indices:
            return np.zeros(len(self.all_pairs), dtype=dtype)
        elif set(self.free_pair_indices) == set(all_indices):
            indices_list = { f:i for i, f in enumerate(self.free_pair_indices)}
            return [values[indices_list[i]] for i in range(len(self.all_pairs))]
        
        if self._stored_solutionchain is None:
            self._stored_solutionchain = self._create_solutionchain()
        return self._stored_solutionchain.solve(values)


    def _solve_all_values_old(self, values: typing.List[float]) -> np.ndarray:
        """solve all the required interaction in two step
        """
        all_indices = range(len(self.all_pairs))
        dtype = self.homogeneous_equation.dtype

        if len(values) != len(self.free_pair_indices):
            raise RuntimeError(
                "{:s}: number of input values of interaction {:d} != required {:d}".format(
                    self.__name__, len(values), len(self.free_pair_indices)
                )
            )
    
        # treat two corner case: no free pair and all pairs are free
        if not self.free_pair_indices:
            return np.zeros(len(self.all_pairs), dtype=dtype)
        elif set(self.free_pair_indices) == set(all_indices):
            indices_list = { f:i for i, f in enumerate(self.free_pair_indices)}
            return [values[indices_list[i]] for i in range(len(self.all_pairs))]
        
        # obtain additional_required_indices
        new_rows = np.zeros(
            (len(self.free_pair_indices), len(self.all_pairs)), dtype=dtype)

        for i, ipair in enumerate(self.free_pair_indices):
            new_rows[i, ipair] = 1.0

        tmp = tools.LinearEquation.from_equation(
            np.vstack([self.homogeneous_equation, new_rows]))
        
        additional_required_indices = tmp.free_variable_indices
        
        if additional_required_indices:
            # obtain values for the additional required index by solving 
            related_indices = set()
            indices = np.array(additional_required_indices)
            for row in self.homogeneous_equation:
                if np.all(np.isclose(row[indices], 0.0, atol = params.ztol)):
                    # a row does not containing any of the no-solvable indices
                    continue
                related_indices |= set(
                    np.argwhere(np.invert(np.isclose(row, 0.0, atol=params.ztol))).flatten()
                )
            
            # AOs that depend on the selected additional_required_index

            solvable_indices = sorted(list(set(range(len(self.all_pairs))) - related_indices))
            for i in self.free_pair_indices:
                assert i in solvable_indices

            map_from_old_indices = {old: new for new, old in enumerate(solvable_indices)}

            solvable_parts = tools.LinearEquation.from_equation(
                self.homogeneous_equation[:, np.array(solvable_indices)]
            )
            free_indices_for_solvable_part = [map_from_old_indices[i] for i in self.free_pair_indices]

            solved_values = tools.solve_matrix(solvable_parts.row_echelon_form, free_indices_for_solvable_part, values)
            solved_aopair_values = {
                self.all_pairs[si]:solved_values[i] for i, si in enumerate(solvable_indices)
            } 

            additional_values = []
            for additional_rquired_i in additional_required_indices:
                pair = self.all_pairs[additional_rquired_i]
                reverse_pair = pair.get_reverse_translated()
                if pair in solved_aopair_values:
                    additional_values.append(solved_aopair_values[pair])
                elif reverse_pair in solved_aopair_values:
                    additional_values.append(np.conjugate(solved_aopair_values[reverse_pair]))
                else:
                    raise RuntimeError("interaction pair not found")

        else:
            additional_values = []

        all_required_indices = self.free_pair_indices + additional_required_indices
        all_required_values = np.hstack([values, np.array(additional_values)])

        return tools.solve_matrix(self.homogeneous_equation, all_required_indices, all_required_values)


    @classmethod
    def from_structure_combinedInteraction(
        cls, cell: np.ndarray, positions: np.ndarray, types: np.ndarray, interaction
    ) -> "OrbitalPropertyRelationship":
        from .solve.interaction import AO
        def convert_AO_to_AtomicOrbital(ao: AO) -> AtomicOrbital:
            t1 = int(ao.translation[0])
            t2 = int(ao.translation[1])
            t3 = int(ao.translation[2])
            return AtomicOrbital(
                    ao.primitive_index, ao.n, ao.l, ao.m, t1, t2, t3
                )
        all_OrbitalPair = [
            OrbitalPair(
                convert_AO_to_AtomicOrbital(aopair.l_AO), 
                convert_AO_to_AtomicOrbital(aopair.r_AO)
            ) for aopair in interaction.all_AOPairs
        ]

        return cls(
            cell, positions, types,
            all_OrbitalPair, interaction.free_indices, interaction.homogeneous_equation
        )


    def get_tightbinding_from_free_parameters(self, 
        free_Hijs: typing.List[float], free_Sijs: typing.Optional[typing.List[float]] = None,
        write_solved_Hijs_filename: typing.Optional[str] = None 
    ):
        """return a tight-binding model from the input parameters"""
        from .properties.tightbinding import TightBindingModel
        from .properties.tightbinding import HijR, SijR, Pindex_lm

        def _convert_AtomicOrbital_to_Pindex_lm(ao: AtomicOrbital) -> Pindex_lm:
            return Pindex_lm(
                pindex = ao.pindex, 
                n = ao.n, l = ao.l, m = ao.m,
                translation = ao.translation
            )

        if len(free_Hijs) != len(self.free_pair_indices):
            print(f'input = {len(free_Hijs)}, required = {len(self.free_pair_indices)}')
            raise RuntimeError("the size of input Hijs/Sijs are different from free parameters")

        all_hijs = self._solve_all_values(free_Hijs)
        HijR_list = []
        for ipair, v, pair in zip(range(len(self.all_pairs)), all_hijs, self.all_pairs):
            HijR_list.append(
                HijR(
                    left = _convert_AtomicOrbital_to_Pindex_lm(pair.l_orbital), 
                    right = _convert_AtomicOrbital_to_Pindex_lm(pair.r_orbital),
                    value = v
                )
            )
            
        if write_solved_Hijs_filename is not None:
            with open(write_solved_Hijs_filename, 'w') as f:
                for v, pair in zip(all_hijs, self.all_pairs):
                    f.write(str(pair) + f" : {v:>50.15f}\n")

        if free_Sijs is None:
            tb = TightBindingModel(self.cell, self.positions, self.types, HijR_list)
        else:
            if len(free_Hijs) != len(free_Sijs):
                raise RuntimeError("the size of input Hijs and Sijs are different")
            all_sijs = self._solve_all_values(free_Sijs)
            SijR_list = []
            for v, pair in zip(all_sijs, self.all_pairs):
                SijR_list.append(
                    SijR(
                        left = _convert_AtomicOrbital_to_Pindex_lm(pair.l_orbital), 
                        right = _convert_AtomicOrbital_to_Pindex_lm(pair.r_orbital),
                        value = v
                    )
                )
            
            tb = TightBindingModel(self.cell, self.positions, self.types, HijR_list, SijR_list)

        return tb


    def print_free_pairs(self) -> None:
        for i, index in enumerate(self.free_pair_indices):
            left = self.all_pairs[index].l_orbital
            right = self.all_pairs[index].r_orbital

            left_symbol = tools.chemical_symbols[self.types[left.pindex]]
            right_symbol = tools.chemical_symbols[self.types[right.pindex]]
            left_position = np.dot(self.cell.T, self.positions[left.pindex] + left.translation)
            right_position = np.dot(self.cell.T, self.positions[right.pindex] + right.translation)
            result = f"  {i+1:>3d} > Pair: "
            rij = right_position - left_position
            result += f"{left_symbol:>2s}-{left.pindex:0>2d} " 
            result += f"{tools.parse_orbital(left.n, left.l, left.m):>7s} -> "
            result += f"{right_symbol:>2s}-{right.pindex:0>2d} "
            result += f"{tools.parse_orbital(right.n, right.l, right.m):>7s} "
            result += "r = ({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
            print(result)


    def to_file(self, filename: str) -> None:
        lines = []
        lines.append("# Lattice vectors (A):")
        for i in range(3):
            lines.append("{:>20.8f}{:>20.8f}{:>20.8f}".format(*self.cell[i]))
        lines.append("")
        lines.append(f"# Number of atoms = {len(self.positions)}")
        lines.append("# Atomic Positions (frac.):")
        for t, p in zip(self.types, self.positions):
            lines.append(
                "{:>3s} {:>20.8f}{:>20.8f}{:>20.8f}".format(tools.chemical_symbols[t], *p)
            )
        lines.append("")
        lines.append(f"# Number of free parameters = {len(self.free_pair_indices)}")
        lines += [f"{i:>6d}" for i in self.free_pair_indices]
        
        lines.append("")
        lines.append("# Number of all AO pairs = {:d}".format(len(self.all_pairs)))
        lines.append("# pindex - n - l - m - translation")
        for pair in self.all_pairs:
            lines.append(str(pair))
        lines.append("")
        lines.append("# Homogeneous_equation; dtype = {:s}".format(
            str(self.homogeneous_equation.dtype))
        )
        lines.append("# shape = {:d} {:d}".format(*self.homogeneous_equation.shape))
        # write for sparse matrix
        for ix, iy in np.argwhere(np.abs(self.homogeneous_equation) > 1e-14):
            lines.append(
                "{:>10d}{:>10d}{:>50.15e}".format(ix, iy, self.homogeneous_equation[ix, iy]))
        #for v in self.homogeneous_equation.flatten():
        #    lines.append(f"{v:>60.15f}") # double precision gives around 16 digits accuracy
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        

    @classmethod
    def from_file(cls, filename: str) -> "OrbitalPropertyRelationship":
        with open(filename, 'r') as f:
            f.readline()
            cell = []
            for i in range(3):
                cell.append(np.array(f.readline().split(), dtype=float))
            cell = np.vstack(cell)

            f.readline()
            natom = int(f.readline().split()[-1])
            f.readline()
            types = []
            pos = []
            for i in range(natom):
                aline = f.readline().split()
                types.append(tools.atomic_numbers[aline[0]])
                pos.append(np.array(aline[1:4], dtype=float))
            pos = np.vstack(pos)
            types = np.array(types, dtype=int)
            f.readline()
            nfree = int(f.readline().split()[-1])
            free_indices = []
            for i in range(nfree):
                free_indices.append(int(f.readline()))

            f.readline()
            npair = int(f.readline().split()[-1])
            f.readline()
            all_ao_pair = []
            for i in range(npair):
                all_ao_pair.append(OrbitalPair.from_str(f.readline()))
            
            f.readline()
            dtype = f.readline().split()[-1]
            shape = [int(p) for p in f.readline().split("=")[-1].split() ]
            matrix = np.zeros(shape, dtype = dtype)

            while aline := f.readline():
                ix, iy, value = aline.split()
                matrix[int(ix), int(iy)] = complex(value)
            #list_value = []
            #for _ in range(shape[0] * shape[1]):
            #    list_value.append(f.readline().strip())
            #matrix = np.array(list_value, dtype=dtype).reshape(shape)

        return cls(
            cell, 
            pos, 
            types, 
            all_ao_pair,
            free_indices,
            matrix
        )
    
