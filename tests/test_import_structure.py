from automaticTB.structure import get_structure_from_cif_file
import os 

structure_cifs = {
    "Si"   : os.path.join("resources", "Si_ICSD51688.cif"),
    "Mg2Sn": os.path.join("resources", "Mg2Sn_mp-2343.cif"),
    "PbTe" : os.path.join("resources", "PbTe_ICSD38295.cif")
}


def test_import_Si():
    filename = structure_cifs["Si"]
    structure = get_structure_from_cif_file(filename)
    for cluster in structure.nnclusters:
        print(cluster)

    assert len(structure.nnclusters) == 4
    distances = [
        "{:>.4f}".format(0.0), 
        "{:>.4f}".format(2.3514884680), 
        "{:>.4f}".format(3.8399645884), 
    ]
    for cluster in structure.nnclusters:
        assert "{:>.4f}".format(cluster.distance) in distances

def test_import_Si_2nn():
    filename = structure_cifs["Si"]
    structure = get_structure_from_cif_file(filename, rcut=3.9)
    for cluster in structure.nnclusters:
        print(cluster)

    assert len(structure.nnclusters) == 6
    distances = [
        "{:>.4f}".format(0.0), 
        "{:>.4f}".format(2.3514884680), 
        "{:>.4f}".format(3.8399645884), 
    ]
    for cluster in structure.nnclusters:
        assert "{:>.4f}".format(cluster.distance) in distances

def test_import_Mg2Sb():
    filename = structure_cifs["Mg2Sn"]
    structure = get_structure_from_cif_file(filename)

    answers = {
        # for Mg, it is tetrahedrally coordinated with 4 nearest neighbor Sn, 
        # further out are 6 Mg as secondary neigbor, so in total 11 atoms
        "Mg": {
            "pointgroup": "-43m", 
            "distances": [
                "{:>.4f}".format(0.0), 
                "{:>.4f}".format(2.95169430), 
                "{:>.4f}".format(3.408323), 
            ]
        },
        # for Sn, 8 Mg coordinate with Sn in cubic fashion
        "Sn": {
            "pointgroup": "m-3m", 
            "distances": [
                "{:>.4f}".format(0.0), 
                "{:>.4f}".format(2.9516943), 
            ]
        }
    }

    for cluster in structure.nnclusters:
        print(cluster)

    for cluster in structure.nnclusters:
        answer = answers[cluster.center_site.site.chemical_symbol]
        assert cluster.sitesymmetrygroup.groupname == answer["pointgroup"]
        assert "{:>.4f}".format(cluster.distance) in answer["distances"]

def test_import_Mg2Sb_rcut():
    filename = structure_cifs["Mg2Sn"]
    structure = get_structure_from_cif_file(filename, rcut = 5.0)

    answers = {
        # for Mg, it is tetrahedrally coordinated with 4 nearest neighbor Sn, 
        # further out are 6 Mg as secondary neigbor, so in total 11 atoms
        "Mg": {
            "pointgroup": "-43m", 
            "distances": [
                "{:>.4f}".format(0.0), 
                "{:>.4f}".format(2.95169430), 
                "{:>.4f}".format(3.408323),  
                "{:>.4f}".format(4.82009661), 
            ]
        },
        # for Sn, 8 Mg coordinate with Sn in cubic fashion
        "Sn": {
            "pointgroup": "m-3m", 
            "distances": [
                "{:>.4f}".format(0.0), 
                "{:>.4f}".format(2.9516943), 
                "{:>.4f}".format(4.82009661), 
            ]
        }
    }

    for cluster in structure.nnclusters:
        print(cluster)

    for cluster in structure.nnclusters:
        answer = answers[cluster.center_site.site.chemical_symbol]
        assert cluster.sitesymmetrygroup.groupname == answer["pointgroup"]
        assert "{:>.4f}".format(cluster.distance) in answer["distances"]

def test_import_PbTe():
    filename = structure_cifs["PbTe"]
    structure = get_structure_from_cif_file(filename)

    answer = {
        "pointgroup": "m-3m", 
        "distances": [
            "{:>.4f}".format(0.0), 
            "{:>.4f}".format(3.231), 
        ]
    }

    for cluster in structure.nnclusters:
        print(cluster)

    assert len(structure.nnclusters) == 4
    for cluster in structure.nnclusters:
        assert cluster.sitesymmetrygroup.groupname == answer["pointgroup"]
        assert "{:>.4f}".format(cluster.distance) in answer["distances"]

test_import_Si()
test_import_Si_2nn()
test_import_Mg2Sb()
test_import_Mg2Sb_rcut()
test_import_PbTe()