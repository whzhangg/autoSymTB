import numpy as np

Hij = np.zeros((7,7))
MOlist = [
    "s", "px", "py", "pz", "s_s", "s_px", "s_py"
]

interactions = {
    "s s": -25,
    "px px": -13,
    "py py": -13,
    "pz pz": -13,
    "s_s s_s": -14,
    "s_px s_px": -12,
    "s_py s_py": -12,
    "s s_s": 3.0,
    "s_s s": 3.0,
    "px s_px": 4.0,
    "s_px px": 4.0,
    "py s_py": 4.0,
    "s_py py": 4.0
}

for i,mo1 in enumerate(MOlist):
    for j,mo2 in enumerate(MOlist):
        Hij[i,j] = interactions.setdefault(" ".join([mo1,mo2]),0)

print(Hij)
w, v = np.linalg.eig(Hij)
print(sorted(w))
