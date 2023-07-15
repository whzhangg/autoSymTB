
def get_HijRs_from_wannier_file(
        filename: str, rcut: float, hcut: float = 0.0) -> typing.List[tb.HijR]:
    """get tight-binding pairs, only real values are taken"""
    zero_translation = np.zeros(3)
    c, p, t = get_Mg3Sb2_cpt()
    all_hijrs = []
    with open(filename,'r') as f:
        f.readline()
        nbnd = int(f.readline().strip())
        numR = int(f.readline().strip())
        tmp_w = []
        while len(tmp_w) < numR:
            datas = f.readline().rstrip().split()
            tmp_w+=datas
        R_w = np.array(tmp_w, dtype=int)  # the weight of R point

        for _ in range(numR * nbnd * nbnd):
            datas = f.readline().rstrip().split()
            value = float(datas[5]) + 1j * float(datas[6])
            if np.abs(value) < hcut: continue
            t = np.array(datas[0:3], dtype=int)
            ibnd = int(datas[3]) - 1
            jbnd = int(datas[4]) - 1
            
            l_state = convert_orbindex_trans_to_plm(ibnd, zero_translation)
            r_state = convert_orbindex_trans_to_plm(jbnd, t)
                    
            dist = get_distance(c, p, t, l_state.pindex, r_state.pindex)
            if dist > rcut: continue
            
            all_hijrs.append(
                    tb.HijR(l_state, r_state, float(datas[5]))
                )
        
    return all_hijrs

def convert_dictionary_to_hijrs(hijrs_dict):
    hijrs = []
    for k, v in hijrs_dict.items():
        part1, part2 = k.split("->")
        lp, ln, ll, lm, lt1, lt2, lt3 = [ int(i) for i in part1.split() ]
        rp, rn, rl, rm, rt1, rt2, rt3 = [ int(i) for i in part2.split() ]

        l_state = tb.Pindex_lm(lp, ln, ll, lm, np.array((lt1, lt2, lt3)))
        r_state = tb.Pindex_lm(rp, rn, rl, rm, np.array((rt1, rt2, rt3)))

        hijrs.append(tb.HijR(l_state, r_state, v))
    return hijrs
    
def convert_hijrs_to_dictionary(hijrs: typing.List[tb.HijR]):
    all_orb_pairs = {}
    for hijr in hijrs:
        
        ref_str1 = "{:>4d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d}".format(
            hijr.left.pindex, hijr.left.n, hijr.left.l, hijr.left.m, int(hijr.left.translation[0]), int(hijr.left.translation[1]), int(hijr.left.translation[2])
        )
        ref_str2 = "{:>4d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d} {:>3d}".format(
            hijr.right.pindex, hijr.right.n, hijr.right.l, hijr.right.m, int(hijr.right.translation[0]), int(hijr.right.translation[1]), int(hijr.right.translation[2])
        )
        all_orb_pairs[f"{ref_str1} -> {ref_str2}"] = hijr.value
    return all_orb_pairs


def get_distance(
    cell: np.ndarray, positions: np.ndarray,
    translation: np.ndarray, iatom: int, jatom: int
) -> float:
    """get distance between two atoms"""
    iatom = positions[iatom]
    jatom = positions[jatom]
    dr = jatom - iatom + translation
    return np.linalg.norm(cell.T @ dr)
