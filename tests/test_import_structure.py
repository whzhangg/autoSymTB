from automaticTB.functions import get_structure_from_cif_file
import os 

structure_cifs = {
    "Si"   : os.path.join("..", "resources", "Si_ICSD51688.cif"),
    "Mg2Sn": os.path.join("..", "resources", "Mg2Sn_mp-2343.cif"),
    "PbTe" : os.path.join("..", "resources", "PbTe_ICSD38295.cif")
}

defined_orbital = {
    "Si" : "3s3p4s",
    "Mg" : "3s",
    "Sn" : "5s5p",
    "Pb" : "6s6p",
    "Te" : "5s5p"
}


def test_import_Si():
    filename = structure_cifs["Si"]
    structure = get_structure_from_cif_file(filename, defined_orbital)

    answers = {
        "pointgroup": "-43m",
        "nsites": 5,
        "norb": 25
    }

    assert len(structure.nnclusters) == 2

    for nncluster in structure.nnclusters:
        assert nncluster.num_sites == answers["nsites"]
        assert nncluster.orbitalslist.num_orb == answers["norb"]
        assert nncluster.sitesymmetrygroup.groupname == answers["pointgroup"]


def test_import_Mg2Sb():
    filename = structure_cifs["Mg2Sn"]
    structure = get_structure_from_cif_file(filename, defined_orbital)

    answers = {
        # for Mg, it is tetrahedrally coordinated with 4 nearest neighbor Sn, 
        # further out are 6 Mg as secondary neigbor, so in total 11 atoms
        "Mg": {"pointgroup": "-43m", "norb": 23, "nsites": 11},
        # for Sn, 8 Mg coordinate with Sn in cubic fashion
        "Sn": {"pointgroup": "m-3m", "norb": 12, "nsites": 9}
    }

    assert len(structure.nnclusters) == 3

    for nncluster in structure.nnclusters:
        ele = nncluster.crystalsites[nncluster.origin_index].site.chemical_symbol
        assert ele in answers
        assert nncluster.num_sites == answers[ele]["nsites"]
        assert nncluster.sitesymmetrygroup.groupname == answers[ele]["pointgroup"]
        assert nncluster.orbitalslist.num_orb == answers[ele]["norb"]

def test_import_PbTe():
    filename = structure_cifs["PbTe"]
    structure = get_structure_from_cif_file(filename, defined_orbital)

    answers = {
        "pointgroup": "m-3m",
        "nsites": 7,
        "norb": 28
    }
    
    assert len(structure.nnclusters) == 2

    for nncluster in structure.nnclusters:
        ele = nncluster.crystalsites[nncluster.origin_index].site.chemical_symbol
        assert ele in ["Pb", "Te"]
        assert nncluster.num_sites == answers["nsites"]
        assert nncluster.sitesymmetrygroup.groupname == answers["pointgroup"]
        assert nncluster.orbitalslist.num_orb == answers["norb"]
