import dataclasses

# operation is one of the 
# -6, -4, -3, m(-2), -1, 1, 2, 3, 4, 6
# direction is one used by Bilbao database, including 000 for identity and inversion
# sense is either + or - or "" (empty)

@dataclasses.dataclass
class SeitzSymbol:
    operation: str
    direction: str 
    sense: str         # +, -, ""
    # sense uniquely tell apart 4+, 4-, 
    @classmethod
    def from_string(cls, string: str):
        parts = string.split("")
        if len(parts) <= 2:
            sense = ""
        else:
            sense = parts[2]
        return cls(parts[0], parts[1], sense)

    def __str__(self):
        to_print = [self.operation, self.direction, self.sense]
        return " ".join([ p for p in to_print if p ])