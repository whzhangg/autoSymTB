# interface to Bilbao
import typing, dataclasses, os, yaml
import numpy as np

current_folder = os.path.split(__file__)[0]

def get_cell_parameter(pointgroup: str) -> typing.Dict[str, np.ndarray]:
    cube = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 
                'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', 
                '4/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
    cubic = np.eye(3)
    hexagonal = np.array([[1.0,0.0,0.0],[-0.5, np.sqrt(3)/2, 0.0],[0.0,0.0,1.0]])
    if pointgroup in cube:
        return cubic
    else:
        return hexagonal


@dataclasses.dataclass
class BilbaoGroupOperation:
    seitz: str
    matrix: np.ndarray
    chi: typing.Dict[str, typing.Union[float, complex]]
    

@dataclasses.dataclass
class BilbaoPointGroup:
    name: str
    operations: typing.List[BilbaoGroupOperation]
    subgroups: typing.List[str]

    @property
    def seitz_operation_dict(self) -> typing.Dict[str, BilbaoGroupOperation]:
        result = {}
        for op in self.operations:
            result[op.seitz] = op
        return result

    def as_dict(self):
        result = {
            "name": self.name,
            "operations": {},
            "subgroups": self.subgroups
            }
        for op in self.operations:
            result["operations"][op.seitz] = {
                "matrix": op.matrix.flatten().tolist(),
                "chi": op.chi
            }
        return result

    @classmethod
    def from_dict(cls, input):
        name = input["name"]
        subgroups = input["subgroups"]
        operations = []
        for seitz, op in input["operations"].items():
            matrix = np.array(op["matrix"]).reshape((3,3))
            operations.append(BilbaoGroupOperation(seitz, matrix, op["chi"]))
        return cls(name, operations, subgroups)


class PointGroupData:
    # this class gathers all data from bilbao
    reference_real_file_yaml = \
        os.path.join(current_folder, "Bilbao", "reference", "real.yaml")
    reference_complex_file_yaml = \
        os.path.join(current_folder, "Bilbao", "reference", "complex.yaml")

    def __init__(self, complex_character: bool, update: bool = False) -> None:
        self.usecomplex = complex_character
        self._datafile = self.reference_complex_file_yaml if self.usecomplex else self.reference_real_file_yaml
        if update or not os.path.exists(self._datafile):
            self._make_datafile()
        self._cached_groups = self._load_datafile_to_dict()
    
    def get_BilbaoPointGroup(self, groupname: str) -> BilbaoPointGroup:
        #if groupname not in self._cached_groups:
        #    self._cached_groups[groupname] = self._make_bilbaopointgroup(groupname)
        return self._cached_groups[groupname]

    def _make_datafile(self):
        directory, f = os.path.split(self._datafile)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        from ..group_list import GroupsList
        from ..utilities import rotation_fraction_to_cartesian
        from .extract_group_characters import get_GroupCharacterInfo_from_bilbao
        from .group_operations import get_group_operation
        from .symmetry_map import mapper_seitz_to_simplified
        
        groupdata = {}
        for groupname in GroupsList:
            charactersInfo = get_GroupCharacterInfo_from_bilbao(groupname, self.usecomplex)
            operations = get_group_operation(groupname)
            lattice = get_cell_parameter(groupname)
            ops = []
            for seitz, matrix in operations.items():
                cartesian_rotation = rotation_fraction_to_cartesian(matrix, lattice)
                simplified_name = mapper_seitz_to_simplified[groupname][seitz]
                ops.append(BilbaoGroupOperation(seitz, cartesian_rotation, charactersInfo.SymWithChi[simplified_name]))
            
                groupdata[groupname] = BilbaoPointGroup(groupname, ops, charactersInfo.subgroups).as_dict()
        
        yaml_str = yaml.dump(groupdata)
        with open(self._datafile, 'w') as f:
            f.write(yaml_str)

    def _load_datafile_to_dict(self) -> typing.Dict[str, BilbaoPointGroup]:
        with open(self._datafile, 'r') as f:
            groupdata = yaml.load(f, Loader=yaml.Loader)
        result = {}
        for key, value in groupdata.items():
            result[key] = BilbaoPointGroup.from_dict(value)
        return result
