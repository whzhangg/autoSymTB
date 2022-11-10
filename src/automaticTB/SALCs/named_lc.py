import dataclasses, typing
from .linear_combination import LinearCombination

@dataclasses.dataclass
class IrrepSymbol:
    symmetry_symbol: str
    main_irrep: str
    main_index: int

    @classmethod
    def from_str(cls, input: str):
        """
        the main symbol ^ index -> subduction symbol
        """
        parts = input.split("->")
        mains = parts[0].split("^")
        main_irrep = mains[0]
        main_index = int(mains[1])
        sym_symbol = "->".join([p.split("^")[0] for p in parts])
        return cls(sym_symbol, main_irrep, main_index)

    def __repr__(self) -> str:
        return f"{self.symmetry_symbol} @ {self.main_index}^th {self.main_irrep}"


class NamedLC(typing.NamedTuple):
    name: IrrepSymbol
    lc: LinearCombination
