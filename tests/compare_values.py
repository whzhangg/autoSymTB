import dataclasses
import os
import typing

import numpy as np

from automaticTB.properties import tightbinding, kpoints
from automaticTB.solve import interaction

current_path = os.path.split(__file__)[0]
data_folder = "compare_values"
stored_value_folder = "stored_interactions"
tol = 1e-3

def compare_stored_values(prefix, tb: tightbinding.TightBindingModel, kpath_str: str):
    unitcell = kpoints.UnitCell(tb.cell)
    kpath = unitcell.get_kpath_from_path_string(kpath_str)
    values = tb.solveE_at_ks(kpath.kpoints)

    SolvedEigenValues.compare_stored_data(prefix, np.hstack([kpath.kpoints, values]))


@dataclasses.dataclass
class SolvedEigenValues:
    """store solved energy with rows like [k0,k1,k2,e0,e1,...]"""
    system_name: str

    @classmethod
    def compare_stored_data(
            cls, name: str, data_compare: np.ndarray) -> "SolvedEigenValues":
        """compare stored data if available, if not, raise error"""
        data = SolvedEigenValues(name)
        filepos = data.file_position
        if os.path.exists(filepos):
            stored_data = np.load(filepos)
            if stored_data.shape != data_compare.shape:
                raise ValueError('stored shape: ', stored_data.shape, 
                                 " and input shape: ", data_compare.shape)
            if not np.allclose(stored_data, data_compare, atol=tol):
                total_number = len(data_compare.flatten())
                total_error = (total_number 
                    - np.sum(np.isclose(data_compare, stored_data, atol = tol).flatten()))
                raise ValueError(f"{total_error} values out of {total_number} are wrong !")
            print("Results the same as stored !!")
        else:
            np.save(data.file_position, data_compare)
            print(f"No stored value for {name}, created !!")
        return data


    def get_data(self) -> np.ndarray:
        return np.load(self.file_position)

    @property
    def file_position(self) -> str:
        return os.path.join(
            current_path, data_folder, "{:s}.npy".format(self.system_name)
        )


@dataclasses.dataclass
class StoredInteractions:
    """find the corresponding values automatically given the pairs"""
    stored_data: typing.Dict[str, complex]

    @classmethod
    def from_system_name(cls, name: str) -> "StoredInteractions":
        filepos = os.path.join(stored_value_folder, f"{name}.stored_interactions.dat")
        interactions = {}
        with open(filepos, 'r') as f:
            while aline := f.readline():
                name, value = aline.split(" : ")
                interactions[name] = complex(value)
        return cls(interactions)

    def find_value(self, pair: interaction.AOPair):
        entry = str(pair)
        return self.stored_data[entry]

if __name__ == "__main__":
    sample_data = np.eye(100)
    SolvedEigenValues.compare_stored_data("test",sample_data)
    wrong_data = np.copy(sample_data)
    wrong_data[5, 20] = -1.0
    SolvedEigenValues.compare_stored_data("test",wrong_data)