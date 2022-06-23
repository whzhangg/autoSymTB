import typing, dataclasses
import matplotlib.pyplot as plt
import numpy as np

@dataclasses.dataclass
class CoordinationResult:
    name: str
    group: str
    nfree: int 
    ntotal: int 

def open_result(filename: str) -> typing.List[CoordinationResult]:
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    result = []
    for aline in lines:
        parts = aline.rstrip().split(",")
        parts = [p.strip() for p in parts]
        result.append(
            CoordinationResult(
                parts[0], parts[1],
                int(parts[2]), int(parts[3])
            )
        )
    return result


def plot_result():
    datas = open_result("free_parameter.dat")
    fig = plt.figure(dpi=300)
    x = np.arange(len(datas))
    y_free = np.array([d.nfree for d in datas])
    y_total = np.array([d.ntotal for d in datas])
    #group = [f"${d.group}$" for d in datas]
    label = [f"({ix+1})" for ix in x]

    axs = fig.add_axes([0.15, 0.3, 0.7, 0.6])
    axs.bar(x, y_free, alpha = 0.6, width = 0.4, color = "red", label = "Independent")
    axs.bar(x, y_total, alpha = 0.1, width = 0.4, color = "blue", label = "Total")

    axs.set_ylim(0, 140)
    axs.set_ylabel("Number of Parameters")

    axs.xaxis.set_major_locator(plt.FixedLocator(x))
    #axs.set_xticklabels(label, rotation = 45, ha="right")
    axs.set_xticklabels(label)
    #axs.xaxis.set_major_formatter(plt.FixedFormatter(group))
    for ix, iy in zip(x, y_free):
        axs.text(ix, iy + 2, str(iy), size = 12, ha = "center", va = "bottom", transform = axs.transData, color = "black")
    #for ix, iy in zip(x, y_total):
    #    axs.text(ix, iy + 2, str(iy), size = 12, ha = "center", va = "bottom", transform = axs.transData, color = "black")
    axs.legend(frameon = False)
    fig.set
    fig.savefig("free_parameter.png")

plot_result()