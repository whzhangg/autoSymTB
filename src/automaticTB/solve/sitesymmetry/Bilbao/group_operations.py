import dataclasses
import numpy as np
import typing, re
from bs4 import BeautifulSoup
from .crawler import Bilbao_PointGroups
from ..seitz_symbol import SeitzSymbol

bilbaopointgroups = Bilbao_PointGroups()

def change_rhombohedral_seitz(input_seitz: str) -> str:
    # database give the trigonal system 312, 3m1 , -31m, with 
    # the symmetry axis given [1-10], [120], [210]
    # but it is more convenient to use [100],[010],[110]
    if "120" in input_seitz:
        return input_seitz.replace("120", "100")
    elif "210" in input_seitz:
        return input_seitz.replace("210", "010")
    elif "1-10" in input_seitz:
        return input_seitz.replace("1-10", "110")
    else:
        return input_seitz

def data_from_soup(soup: BeautifulSoup):
    name_header = soup.find("h2")
    parts = re.findall(r"\(\S+\)", name_header.text)  # (422)
    group_name = parts[0][1:-1]
    main_table = soup.find_all("table",{"border": "5", "cellpadding": "10"})
    table = main_table[0]
    seitz_and_matrix = {}
    for row in table.children:
        if row.name != "tr":
            continue
            
        row_contents = []
        for cell in row.children:
            if cell.name == "td":
                row_contents.append(re.findall(r"\S+",cell.text))
            
        try:
            current = int(row_contents[0][0])
            assert current == len(seitz_and_matrix.items()) + 1
            operation = np.array(row_contents[2],dtype = float).reshape((3,3))
            seitz_op = row_contents[-1][0]
            seitz_dir = ""
            seitz_sense = ""

            if len(row_contents[-1]) == 1:
                seitz_dir = "000"
                seitz_sense = ""
            elif len(row_contents[-1]) == 2:
                seitz_sense = ""
                seitz_dir = row_contents[-1][1]
            elif len(row_contents[-1]) == 3:
                seitz_sense = row_contents[-1][1]
                seitz_dir = row_contents[-1][2]
                    
            seitz = SeitzSymbol(seitz_op, seitz_dir, seitz_sense)
            seitz_and_matrix[str(seitz)] = operation
        except ValueError:
            # when the first element of the row is not a index 
            pass

    return group_name, seitz_and_matrix

def get_group_operation(groupname: str) -> typing.Dict[str, np.ndarray]:
    # return a dictionary from seitz to matrix
    soup = bilbaopointgroups.get_group_operation_page_as_soup(groupname)
    designated_groupname, seitz_matrix = data_from_soup(soup)

    if designated_groupname == groupname:
        return seitz_matrix
    
    # only 312, 3m1, -31m
    if designated_groupname == "3m1":
        return seitz_matrix
    else:
        replaced = {}
        for s, m in seitz_matrix.items():
            replaced[change_rhombohedral_seitz(s)] = m
        return replaced


        

