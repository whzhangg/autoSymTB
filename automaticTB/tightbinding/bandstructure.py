import typing, dataclasses
from .model import TightBindingBase
from .kpoints import Kpath, BandPathTick
import numpy as np

@dataclasses.dataclass
class BandStructureResult:
    x: np.ndarray
    E: np.ndarray
    ticks: typing.List[BandPathTick]

    def write_data_to_file(self, filename: str):
        if len(self.x) != self.E.shape[0]:
            raise "Size of energy not equal to x positions"
        ticks = [ "{:>s}:{:>10.6f}".format(tick.symbol, tick.xpos) for tick in self.ticks]
        results = " & ".join(ticks) + "\n"
        results += f"nk   {self.E.shape[0]}\n"
        results += f"nbnd {self.E.shape[1]}\n"
        dataformat = "{:>20.12e}"
        for x, ys in zip(self.x, self.E):
            results += dataformat.format(x)
            for y in ys:
                results += dataformat.format(y)
            results += "\n"
        with open(filename, 'w') as f:
            f.write(results)


    def __eq__(self, o: object) -> bool:
        return np.allclose(self.x, o.x) and \
               np.allclose(self.E, o.E) and \
                   self.ticks == o.ticks


    @classmethod
    def from_datafile(cls, filename: str):
        with open(filename, 'r') as f:
            aline = f.readline()
            parts = aline.split("&")
            ticks = [
                BandPathTick(part.split(":")[0].strip(), float(part.split(":")[1])) for part in parts
            ]
            nk = int(f.readline().split()[-1])
            nbnd = int(f.readline().split()[-1])
            xy = []
            for ik in range(nk):
                xy.append( 
                    np.array(f.readline().split(), dtype=float) 
                )
            xy = np.vstack(xy)
        return cls(
            xy[:,0],
            xy[:,1:],
            ticks
        )


    def plot_data(self, filename: str):
        # plot simple band structure
        import matplotlib.pyplot as plt
        fig = plt.figure()
        axes = fig.subplots()

        axes.set_ylabel("Energy (eV)")
        axes.set_xlabel("High Symmetry Point")
        ymin = np.min(self.E)
        ymax = np.max(self.E)
        ymin = ymin - (ymax - ymin) * 0.05
        ymax = ymax + (ymax - ymin) * 0.05

        axes.set_xlim(min(self.x), max(self.x))
        axes.set_ylim(ymin, ymax)

        for ibnd in range(self.E.shape[1]):
            axes.plot(self.x, self.E[:, ibnd])
        for tick in self.ticks:
            x = tick.xpos
            axes.plot([x,x], [ymin,ymax], color='gray')

        tick_x = [ tick.xpos for tick in self.ticks ]
        tick_s = [ tick.symbol for tick in self.ticks ]
        axes.xaxis.set_major_locator(plt.FixedLocator(tick_x))
        axes.xaxis.set_major_formatter(plt.FixedFormatter(tick_s))

        fig.savefig(filename)


def get_bandstructure_result(
    tb: TightBindingBase, kpath: Kpath
) -> BandStructureResult:
    energies = tb.solveE_at_ks(kpath.kpoints)
    return BandStructureResult(
        kpath.xpos, energies, kpath.ticks
    )