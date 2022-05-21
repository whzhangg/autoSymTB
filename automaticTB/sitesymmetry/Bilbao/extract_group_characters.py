import typing, re, math, bs4, dataclasses
from .crawler import Bilbao_PointGroups

bilbaopointgroups = Bilbao_PointGroups()

w = math.cos(2*math.pi/3) + math.sin(2*math.pi/3) * 1.0j
w_dict = {
    "w": w,
    "w2": w**2,
    "-w": -1.0 * w,
    "-w2": -1.0 * w**2
}

def get_table_title(table: bs4.element.Tag) -> str:
    # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#strings-and-stripped-strings
    return next(table.caption.stripped_strings)

def get_tables(soup: bs4.BeautifulSoup) -> typing.Dict[str, bs4.element.Tag]:
    tables = soup.find_all("table")
    table_dict = {}
    for table in tables:
        title = get_table_title(table) if table.caption else None
        if title:
            table_dict[title] = table
    return table_dict

def get_subgroup_names_from_table(table: bs4.element.Tag) -> typing.List[str]:
    subgroups: typing.List[str] = []
    for ir,row in enumerate(table.find_all("tr")):
        if ir == 0: continue  
        # the first row is header
        for last in row.td.stripped_strings:
            pass
        bracketed = re.findall(r"\(\S+\)", last)[0]
        subgroups.append(bracketed[1:-1])
    return subgroups

def parse_table_operation(row: typing.List[bs4.element.Tag]) -> typing.List[str]:
    op_name = []
    for cell in row[1:]:
        op_name.append("_".join([s for s in cell.stripped_strings]))
    return op_name

def cell_has_br_tag(cell) -> bool:
    has_br = False
    for d in cell.descendants:
        if d.name == "br": 
            has_br = True
            break
    return has_br

def get_stripped_strings_separated_by_a_br(cell
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    splitted = str(cell).split("<br/>")
    assert len(splitted) == 2
    texts = re.sub(r"<.+?>", " ", splitted[0])
    upper = texts.split()
    texts = re.sub(r"<.+?>", " ", splitted[1])
    lower = texts.split()
    return (upper, lower)

def get_complex_value_from_string(input: str) -> complex:
    if "w" in input:
        return w_dict[input]
    else:
        return complex(input)

def parse_representation(row: typing.List[bs4.element.Tag]
) -> typing.Dict[str, typing.List[typing.Union[float, complex]]]:
    is_multiline = cell_has_br_tag(row[0])

    if is_multiline:
        character_values = []
        tops, bottoms = get_stripped_strings_separated_by_a_br(row[0])
        rep_up = ["_".join(tops)]
        rep_dw = ["_".join(bottoms)]
        
        for cell in row[1:]:
            tops, bottoms = get_stripped_strings_separated_by_a_br(cell)
            rep_up.append( get_complex_value_from_string("".join(tops)) )
            rep_dw.append( get_complex_value_from_string("".join(bottoms)) )
        return {
            rep_up[0]: rep_up[1:],
            rep_dw[0]: rep_dw[1:],
        }

    if not is_multiline:
        name = "_".join([s for s in row[0].stripped_strings])
        characters = []
        for cell in row[1:]:
            character_values = [s for s in cell.stripped_strings]
            if len(character_values) == 1:
                characters.append(complex(character_values[0]))
        return {name: characters}

def get_operation_character_from_table(table: bs4.element.Tag, use_complex: bool):
    # the first row is operation name
    # the second row is multiplicity, which is not needed
    table_content = []
    for ir,row in enumerate(table.find_all("tr")):
        row_content = []
        is_multiplicity_line = False
        for cell in row.find_all("td"):
            if "Mult" in cell.text: 
                is_multiplicity_line = True
                break
            row_content.append(cell)
        if is_multiplicity_line:
            continue

        row_content.pop(1)  # we pop the schoenflies notation, which exist in all case
        table_content.append(row_content)
    
    contain_function = "functions" in table_content[0][-1].text
    if contain_function:
        for row in table_content:
            row.pop(-1)
    
    operations = parse_table_operation(table_content[0])
    representations = {}
    for row in table_content[1:]:
        RepChi_dict = parse_representation(row)
        representations.update(RepChi_dict)
    
    if use_complex: 
        return operations, representations
        
    # combine the same characters
    new = {}
    for key, value in representations.items():
        if key[0:2] == "1_":
            other = key.replace("1_", "2_")
            new_representation = key.lstrip("1_")
            new_value = [ (w1+w2).real for w1,w2 in zip(value, representations[other]) ]
            new[new_representation] = new_value
        elif key[0:2] == "2_":
            other = key.replace("2_", "1_")
            new_representation = key.lstrip("2_")
            new_value = [ (w1+w2).real for w1,w2 in zip(value, representations[other]) ]
            new[new_representation] = new_value
        else:
            new[key] = [ v.real for v in value ]
    return operations, new

def get_name_characters_from_line(
        aline:str
) -> typing.Tuple[str, typing.List[typing.Union[complex, float]]]:
    parts = aline.split()
    rep_name = parts[0]

    try:
        characters = [ float(p) for p in parts[1:] ]
    except ValueError:
        characters = [ complex(p) for p in parts[1:]]
    return rep_name, characters

@dataclasses.dataclass
class GroupCharacterInfo:
    name: str
    SymWithChi: typing.Dict[str, typing.Dict[str, typing.Union[float, complex]]]
    subgroups: typing.List[str]

    @property
    def character_is_complex(self) -> bool:
        each_character_is_complex: typing.List[bool] = []
        for chis in self.SymWithChi.values():
            each_character_is_complex += [
                type(c) == complex for c in chis.values()
            ]
        assert len(set(each_character_is_complex)) == 1, self.name
        return any(each_character_is_complex)

    @classmethod
    def from_file(cls, fn: str):
        with open(fn, "r") as f:
            lines = f.readlines()
        
        group_name = lines[0].split("=")[-1].rstrip().lstrip()
        
        syms_line = lines[3].rstrip().split()
        symmetry = { op:{} for iop, op in enumerate(syms_line) if iop % 2 == 0 }
        symmetrynames = list(symmetry.keys())
        irrep_names = []
        for aline in lines[4:-3]:
            rep_name, characters = get_name_characters_from_line(aline)
            irrep_names.append(rep_name)
            for isym, sym in enumerate(symmetrynames):
                symmetry[sym][rep_name] = characters[isym]
    
        separated_subgroups = lines[-2].split("=")[-1]
        subgroups = [ sgn.lstrip().rstrip() for sgn in separated_subgroups.split(",")]

        return cls(group_name, symmetry, subgroups)
    
    def write_to_file(self, fn:str):
        table_rows = []
        table_rows.append([""] + [op for op in self.SymWithChi.keys()])
        for rep in self.SymWithChi["1"].keys():
            row = [rep]
            for value in self.SymWithChi.values():
                row.append(value[rep])
            table_rows.append(row)
        

        n_operations = len(self.SymWithChi)
        results = f"Point group = {self.name}\n"
        results+= "=" * (6 + 20*n_operations) + "\n"

        for irow, row in enumerate(table_rows):
            if irow == 0:
                formatter = "{:>6s}" + "{:>20.6s}" * n_operations + "\n"
            else:
                formatter = "{:>6s}" + "{:>20.6f}" * n_operations + "\n"
            results += formatter.format(*row)

        results+= "=" * (6 + 20*n_operations) + "\n"
        results+= f"Subgroups = {self.subgroups[0]}"
        for subgroup in self.subgroups[1:]:
            results += ", " + subgroup
        with open(fn, 'w') as f:
            f.write(results)


def get_GroupCharacterInfo_from_bilbao(groupname: str, use_complex: bool) -> GroupCharacterInfo:
    soup = bilbaopointgroups.get_group_main_page_as_soup(groupname)
    t = get_tables(soup)
    subgroup = t["Subgroups of the group"]
    subgroups = get_subgroup_names_from_table(subgroup)

    ops, irreps = get_operation_character_from_table(t["Character Table of the group"], use_complex)
    symwithChi = {}
    for iop, op in enumerate(ops):
        opcharacter = {}
        for irrepname, character in irreps.items():
            opcharacter[irrepname] = character[iop]
        symwithChi[op] = opcharacter

    return GroupCharacterInfo(groupname, symwithChi, subgroups)
