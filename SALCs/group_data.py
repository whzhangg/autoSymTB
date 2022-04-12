
from symmetry import Symmetry
import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IrreducibleRP:
    symbol: str
    dimension: int
    characters: List[float]

@dataclass
class GroupInfo:
    name_Schoenflies: str
    name_HM_short: str
    order: int
    symmetry_index: List[int]
    Irreps: List[IrreducibleRP]
    subgroup_Schoenflies: Optional[str]

Possible_symmetry = [
    Symmetry((0,       0,            0), 0), #0 E        identity
    Symmetry((0,       0,   torch.pi/3), 1), #3 S3_1     
    Symmetry((0,       0, 2*torch.pi/3), 0), #2 C3_1       120degree rotation
    Symmetry((0,       0,     torch.pi), 1), #1 Sigma_h  horizontal refection
    Symmetry((0,       0, 4*torch.pi/3), 0), #2 C3       120degree rotation    
    Symmetry((0,       0, 5*torch.pi/3), 1), #3 S3_2    
    Symmetry((            0, torch.pi,             0), 0), #4 C2'_1      rotation around x 
    Symmetry(( 2*torch.pi/3, torch.pi, -2*torch.pi/3), 0), #4 C2'_2       
    Symmetry((-2*torch.pi/3, torch.pi,  2*torch.pi/3), 0), #4 C2'_3      
    Symmetry((            0, torch.pi, -torch.pi), 1),  #5 Sigma_v  vertical reflection
    Symmetry((-2*torch.pi/3, torch.pi, -torch.pi), 1),  #5 Sigma_v  vertical reflection
    Symmetry(( 2*torch.pi/3, torch.pi,  torch.pi), 1),   #5 Sigma_v  vertical reflection
]

groupData = {
    "D3h": GroupInfo("D3h", "-6m2", 12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
    Irreps=[
        # E,S3, C3, sigma_h, C3, S3, C2, C2, C3, Sigma_v, Sigma_v, Sigma_v
        IrreducibleRP("A1\'", 1, [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        IrreducibleRP("A2\'", 1, [ 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1]),
        IrreducibleRP("E\'",  2, [ 2,-1,-1, 2,-1,-1, 0, 0, 0, 0, 0, 0]),
        IrreducibleRP("A1\"", 1, [ 1,-1, 1,-1, 1,-1, 1, 1, 1,-1,-1,-1]),
        IrreducibleRP("A2\"", 1, [ 1,-1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1]),
        IrreducibleRP("E\"",  2, [ 2, 1,-1,-2,-1, 1, 0, 0, 0, 0, 0, 0])
    ], subgroup_Schoenflies="Cs"),

    "Cs" : GroupInfo("Cs", "m", 2, [0,9], 
    Irreps=[IrreducibleRP("A\'", 1, [1, 1]), IrreducibleRP("A\"", 1, [1,-1])], subgroup_Schoenflies= "1"),

    "1" : GroupInfo("1", "1", 1, [0], Irreps=[IrreducibleRP("A\'", 1, [1])], subgroup_Schoenflies=None )
}