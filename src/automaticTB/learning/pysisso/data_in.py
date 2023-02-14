"""Module containing classes to create and manipulate SISSO input files."""

import typing

import numpy as np
import pandas as pd  # type: ignore


class SISSOSingleRegDat:
    """Main class containing the data for SISSO
    
    write input data to file / read data from file
    """
    def __init__(self, yX: pd.DataFrame) -> None:
        self.data = yX
        if self.data.index.name is None:
            self.data.index.name = "observations"


    @property
    def nsample(self) -> int:
        """Return number of samples in this data set."""
        return len(self.data)


    @property
    def nsf(self) -> int:
        """Return number of (scalar) features in this data set."""
        return len(self.data.columns) - 1
    
    @property
    def target(self) -> pd.DataFrame:
        return self.data.iloc[:,0]

    @property
    def features(self) -> pd.DataFrame:
        return self.data.iloc[:,1:]

    @property
    def input_string(self) -> str:
        """Input string of the .dat file."""
        out = [
            " ".join(["{:20}".format(column_name) for column_name in self.data.columns])
        ]
        max_str_size = max(self.data[self.data.columns[0]].apply(len))
        header_row_format_str = "{{:{}}}".format(max(20, max_str_size))
        for _, row in self.data.iterrows():
            row_list = list(row)
            line = [header_row_format_str.format(row_list[0])]
            # line = ['{:20}'.format(row_list[0])]
            for col in row_list[1:]:
                line.append("{:<20.12f}".format(col))
            out.append(" ".join(line))
        return "\n".join(out)

    def to_file(self, filename="train.dat") -> None:
        """Write this SISSODat object to file."""
        max_str_size = max(len(i) for i in self.data.index)
        length = max(20, max_str_size)
        nfeature = self.nsf
        title_format = "{{:{}}}".format(length) * (nfeature + 2)
        value_format = "{{:{}}}".format(length) + ("{{:<{}.10e}}".format(length))*(nfeature+1)
        lines = [title_format.format(self.data.index.name, *(c for c in self.data.columns))]
        for index, row in self.data.iterrows():
            lines.append(value_format.format(index, *row.values))

        with open(filename, "w") as f:
            f.write("\n".join(lines))


    @classmethod
    def from_file(cls, filepath: str) -> "SISSOSingleRegDat":
        """Construct SISSODat object from file."""
        if filepath.endswith(".dat"):
            data = pd.read_csv(filepath, delim_whitespace=True, index_col=0)
            return cls(yX=data)
        else:
            raise ValueError("The from_file method is working only with .dat files")


class SISSOUnitIn:
    """handle unit input"""
    def __init__(self,
        feature_unit: typing.List[typing.List[float]],
        target_unit: typing.Optional[typing.List[float]] = None
    ) -> None:
        if target_unit is not None and len(target_unit) != feature_unit.shape[1]:
            raise ValueError(f"target unit different from feature unit dimension")
        self._feature_unit = np.array(feature_unit, dtype=float)
        if target_unit is not None:
            self._target_unit = np.array(target_unit, dtype=float)
        else:
            self._target_unit = None
    
    def __repr__(self) -> str:
        if self._target_unit is None:
            return f"{self.nsf} features with {self.ndim} unit dimension"
        else:
            return f"{self.nsf} features with {self.ndim} unit dimension with target unit"


    @property
    def ndim(self) -> int:
        return self._feature_unit.shape[1]

    @property
    def nsf(self) -> int:
        return len(self._feature_unit)

    def to_file(self, filename: str = "feature_units") -> None:
        formatter = r"{:>10.3f}" * self.ndim + "\n"
        with open(filename, 'w') as f:
            if self._target_unit is not None:
                f.write(formatter.format(*self._target_unit))
            for u in self._feature_unit:
                f.write(formatter.format(*u))
    
    @classmethod
    def from_file(cls, filename: str, nsf: int = None) -> "SISSOUnitIn":
        data = np.loadtxt(filename, dtype=float)
        if nsf is None:
            return cls(feature_unit = data)
        elif nsf == data.shape[0]:
            return cls(feature_unit = data)
        elif nsf == data.shape[0] - 1:
            return cls(feature_unit = data[1:], target_unit = data[0])
        else:
            raise ValueError(f"Input value of nsf = {nsf} is probably wrong")

