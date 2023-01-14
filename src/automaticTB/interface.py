import numpy as np, typing, dataclasses

from .solve.structure import Structure
from .solve.interaction import CombinedAOSubspaceInteraction
from .parameters import tolerance_structure
from .properties import ElectronicModel
from .tools import atomic_numbers, chemical_symbols, parse_orbital, timefn

@dataclasses.dataclass
class AtomicOrbital:
    pindex: int
    n: int 
    l: int
    m: int
    translation: np.ndarray

    def as_line(self) -> str:
        return "{:>3d}{:>3d}{:>3d}{:>3d} t = {:>3d}{:>3d}{:>3d}".format(
            self.p_index, self.n, self.l, self.m, *self.translation
        )

    @classmethod
    def from_line(cls, aline) -> "AtomicOrbital":
        indices, translation = aline.split("t =") 
        idx = indices.split()
        trans = np.array(translation.split(), dtype=int)
        return cls(
            int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]), 
            trans
        )

    def __eq__(self, o: "AtomicOrbital") -> bool:
        return self.pindex == o.pindex and \
               self.n == o.n and self.l == o.l and self.m == o.m and \
               np.allclose(self.translation, o.translation, atol = tolerance_structure)

    def __repr__(self) -> str:
        return "{:>4d} {:>3d} {:>3d} {:>3d} {:>4d} {:>4d} {:>4d}".format(
            self.pindex, self.n, self.l, self.m, *self.translation
        )

    @classmethod
    def from_str(cls, string: str) -> "AtomicOrbital":
        parts = [int(p) for p in string.split()]
        if len(parts) != 7:
            raise RuntimeError("Initializing atomic orbital but number of integer not correct")
        
        return cls(
            parts[0], 
            parts[1], 
            parts[2], 
            parts[3], 
            np.array([parts[4],parts[5],parts[6]], dtype = int)
        )



@dataclasses.dataclass
class OrbitalPair:
    l_orbital: AtomicOrbital
    r_orbital: AtomicOrbital

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
class OrbitalPropertyRelationship:
    """
    this class is the interface between the solve class and the tightbinding and property class
    """
    cell: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    all_pairs: typing.List[OrbitalPair]
    free_pair_indices: typing.List[int]
    homogeneous_equation: np.ndarray

    def _solve_all_values(self, input_values: typing.List[float]) -> np.ndarray:
        """
        similar to mathLA.LinearEquation.solve_providing_values()
        """
        nrow, ncol = self.homogeneous_equation.shape
        nfree = ncol - nrow
        if nfree != len(input_values):
            raise RuntimeError(
                "the input values do not fit with row_echelon_form with shape {:d},{:d}".format\
                (*self.homogeneous_equation.shape)
            )

        data_type = self.homogeneous_equation.dtype
        new_rows = np.zeros((nfree, ncol), dtype=data_type)
        for i, fi in enumerate(self.free_pair_indices):
            new_rows[i, fi] = 1.0
        
        stacked_left = np.vstack([self.homogeneous_equation, new_rows])
        stacked_right = np.hstack(
            [ np.zeros(nrow, dtype=data_type),
              np.array(input_values, dtype=data_type) ]
        )
        return np.dot(np.linalg.inv(stacked_left), stacked_right)


    @classmethod
    def from_structure_combinedInteraction(
        cls, cell: np.ndarray, positions: np.ndarray, types: np.ndarray,
        equation: CombinedAOSubspaceInteraction
    ) -> "OrbitalPropertyRelationship":
        from .solve.interaction import AO

        def convert_AO_to_AtomicOrbital(ao: AO) -> AtomicOrbital:
            return AtomicOrbital(
                ao.primitive_index, ao.n, ao.l, ao.m, np.array(ao.translation, dtype=int)
            )

        all_OrbitalPair = [
            OrbitalPair(
                convert_AO_to_AtomicOrbital(aopair.l_AO), 
                convert_AO_to_AtomicOrbital(aopair.r_AO)
            ) for aopair in equation.all_AOpairs
        ]
        
        free_OrbitalPair = [
            OrbitalPair(
                convert_AO_to_AtomicOrbital(aopair.l_AO), 
                convert_AO_to_AtomicOrbital(aopair.r_AO)
            ) for aopair in equation.free_AOpairs
        ]

        free_indices = sorted([all_OrbitalPair.index(p) for p in free_OrbitalPair])  
        homogeneous_equation = equation.homogeneous_equation.row_echelon_form

        return cls(
            cell, positions, types,
            all_OrbitalPair, free_indices, homogeneous_equation
        )


    def get_ElectronicModel_from_free_parameters(self, 
        free_Hijs: typing.List[float], free_Sijs: typing.Optional[typing.List[float]] = None,
        print_solved_Hijs: bool = False
    ) -> "ElectronicModel":
        from .properties.tightbinding.tightbinding_optimized import TightBindingModelOptimized
        from .properties.tightbinding import HijR, SijR, Pindex_lm

        def _convert_AtomicOrbital_to_Pindex_lm(ao: AtomicOrbital) -> Pindex_lm:
            return Pindex_lm(
                pindex = ao.pindex, 
                n = ao.n, l = ao.l, m = ao.m,
                translation = ao.translation
            )

        if len(free_Hijs) != len(self.free_pair_indices):
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
            if print_solved_Hijs:
                left = pair.l_orbital
                right = pair.r_orbital

                left_symbol = chemical_symbols[self.types[left.pindex]]
                right_symbol = chemical_symbols[self.types[right.pindex]]
                left_position = np.dot(self.cell.T, self.positions[left.pindex] + left.translation)
                right_position = np.dot(self.cell.T, self.positions[right.pindex] + right.translation)
                result = f"> Pair: "
                rij = right_position - left_position
                result += f"{left_symbol:>2s}-{left.pindex:0>2d} " 
                result += f"{parse_orbital(left.n, left.l, left.m):>7s} -> "
                result += f"{right_symbol:>2s}-{right.pindex:0>2d} "
                result += f"{parse_orbital(right.n, right.l, right.m):>7s} "
                result += "r = ({:>6.2f},{:>6.2f},{:>6.2f})".format(*rij)
                result += f" H = {v.real:>10.6f}"
                print(result)

        if free_Sijs is None:
            tb = TightBindingModelOptimized(self.cell, self.positions, self.types, HijR_list)
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
            
            tb = TightBindingModelOptimized(self.cell, self.positions, self.types, HijR_list, SijR_list)

        return ElectronicModel(tbmodel = tb)


    def print_free_pairs(self) -> None:
        for i, index in enumerate(self.free_pair_indices):
            left = self.all_pairs[index].l_orbital
            right = self.all_pairs[index].r_orbital

            left_symbol = chemical_symbols[self.types[left.pindex]]
            right_symbol = chemical_symbols[self.types[right.pindex]]
            left_position = np.dot(self.cell.T, self.positions[left.pindex] + left.translation)
            right_position = np.dot(self.cell.T, self.positions[right.pindex] + right.translation)
            result = f"{i+1:>3d} > Pair: "
            rij = right_position - left_position
            result += f"{left_symbol:>2s}-{left.pindex:0>2d} " 
            result += f"{parse_orbital(left.n, left.l, left.m):>7s} -> "
            result += f"{right_symbol:>2s}-{right.pindex:0>2d} "
            result += f"{parse_orbital(right.n, right.l, right.m):>7s} "
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
                "{:>3s} {:>20.8f}{:>20.8f}{:>20.8f}".format(chemical_symbols[t], *p)
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
        for v in self.homogeneous_equation.flatten():
            lines.append(f"{v:>40.10f}")
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
                types.append(atomic_numbers[aline[0]])
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
            list_value = []
            for _ in range(shape[0] * shape[1]):
                list_value.append(f.readline().strip())
            matrix = np.array(list_value, dtype=dtype).reshape(shape)

        return cls(
            cell, 
            pos, 
            types, 
            all_ao_pair,
            free_indices,
            matrix
        )



