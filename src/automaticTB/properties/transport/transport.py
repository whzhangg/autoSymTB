import numpy as np, typing, dataclasses
from ..tightbinding import TightBindingModel
from ..kpoints import UnitCell
from scipy import constants

def fd(ei,u,T):
    """
    calculate fermi-dirac distribution,
    units of input: ei and u in eV, T in K
    """
    nu = (ei-u) * constants.elementary_charge / (constants.Boltzmann*T)
    return 1/( 1+np.exp( nu ) )

def dfde(ei, u, T):
    """
    SI units
    """
    kbt = constants.Boltzmann * T
    nu = (ei-u)* constants.elementary_charge / kbt
    return  -1.0 * np.exp(nu) / (1.0 + np.exp(nu))**2 / kbt

@dataclasses.dataclass
class TransportResult:
    temperatures: np.ndarray   # ntemp
    mus: np.ndarray            # n_mu
    ncarrier: np.ndarray       # ntemp, n_mu
    seebeck: np.ndarray        # ntemp, n_mu, 3, 3
    conductivity: np.ndarray   # ntemp, n_mu, 3, 3
    ekappa: np.ndarray         # ntemp, n_mu, 3, 3
    tau: float

    @classmethod
    def calculate_from_tightbinding(cls, 
        tb: TightBindingModel, t_range: np.arange, mu_range: np.ndarray, 
        valence: int, 
        tau: float = 1.0,
        nkgrid: int = 20, weight: float = 2.0
    ) -> "TransportResult":
        calculator = TransportCalculator(tb, nkgrid, weight)
        ntemp = len(t_range)
        nmu = len(mu_range)
        ncarrier = np.zeros((ntemp, nmu))
        seebeck = np.zeros((ntemp, nmu, 3, 3))
        sigma = np.zeros((ntemp, nmu, 3, 3))
        ekappa = np.zeros((ntemp, nmu, 3, 3))

        for itemp, temp in enumerate(t_range):
            for imu, mu in enumerate(mu_range):
                ncarrier[itemp, imu] = calculator.calculate_ncarrier(mu, temp, valence)
                tau_shape = (calculator.nk, calculator.nbnd)
                sigma_ij, Sij, k_ij = calculator.calculate_transport(mu, temp, tau * np.ones(tau_shape))
                seebeck[itemp, imu] = Sij
                sigma[itemp, imu] = sigma_ij
                ekappa[itemp, imu] = k_ij

        return cls(
            t_range, mu_range, ncarrier, seebeck, sigma, ekappa, tau
        )

    def mobility_effective_mass(self) -> np.ndarray:
        inv_mass = np.zeros_like(self.conductivity)
        for itemp, temp in enumerate(self.temperatures):
            for imu, mu in enumerate(self.mus):
                inv_mass[itemp, imu] = self.conductivity[itemp, imu] / self.ncarrier[itemp, imu] / self.tau / constants.elementary_charge ** 2 * constants.electron_mass
        
        return inv_mass

    def seebeck_effective_mass(self) -> np.ndarray:
        seebeck_mass = np.zeros_like(self.ncarrier)
        kb = constants.Boltzmann; e = constants.elementary_charge
        h = constants.h
        pi = np.pi 
        for itemp, temp in enumerate(self.temperatures):
            for imu, mu in enumerate(self.mus):
                n = np.abs(self.ncarrier[itemp, imu])
                s = np.abs(np.trace(self.seebeck[itemp, imu]) / 3.0 )
                #print(f"s = {s*1e6:>.8f} n = {n/1e25:>.8e}")
                factor1 = h**2 / (2 * kb * temp)
                factor2 = (3 * (n/1.12) / (16 * np.sqrt(pi)))**(2/3)
                term3_up = (np.exp(s/(kb/e)-2) - 0.17)**(2/3)
                term3_dw = 1 + np.exp(-5 * (s/(kb/e) - (kb/e)/s))
                term4_up = (3/pi**2) * (2/np.sqrt(pi))**(2/3) * (s/(kb/e))
                term4_dw = 1 + np.exp( 5 * (s/(kb/e) - (kb/e)/s))
                seebeck_mass[itemp, imu] = 0.657 * factor1 * factor2 * ( term3_up / term3_dw + term4_up / term4_dw ) / constants.electron_mass
                #print(f"seebeck mass = {seebeck_mass[itemp, imu]}")
        
        return seebeck_mass

class TransportCalculator:
    def __init__(
        self, tbmodel: TightBindingModel, approximate_k: int, weight: float = 2.0
    ) -> None:
        self.tbmodel = tbmodel
        unitcell = UnitCell(self.tbmodel.cell)
        grid = unitcell.recommend_kgrid(approximate_k)
        self.kmesh = unitcell.get_Kmesh(grid)

        self.ek, self.vk = self.tbmodel.solveE_V_at_ks(self.kmesh.kpoints)

        self.nk = self.kmesh.numk
        self.nbnd = len(self.tbmodel.basis)
        self.cell_volume = unitcell.get_volume() # A

        self.weight = weight

    def calculate_ncarrier(self, mu: float, temp: float, nele: float) -> np.ndarray:
        r"""
        mu in eV, temp in K, nele is the reference electron count in the case of semiconductor, 
        nele = 0 will give the total number of electron in the system.
        Positive number means electrons as carrier, negative number means holes

        \frac{1}{N_kV}\sum_{n,k} f(e_{n,k}, \mu, T)
        """
        summed_carrier = np.sum(fd(self.ek, mu, temp)) * self.weight / self.nk
        factor = 1.0 / (self.cell_volume * 1e-30)
        return (summed_carrier - nele) * factor

    def calculate_transport(self, mu: float, temp: float, tau: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """calculates the electrical conductivity, seebeck and kappa in SI units
        require relaxation time the same size as energy, in unit of sec. """
        if tau.shape != self.ek.shape:
            print(f"relaxation time has shape {tau.shape}, " + \
                  f"it should be the same as ek's shape {self.ek.shape}")
            raise RuntimeError

        vel_tensor = np.einsum("mni, mnj -> mnij", self.vk, self.vk)
        
        dfde_factor = -1 * dfde(self.ek, mu, temp)
        
        sigma_sum = np.zeros((3,3))
        sigmaSsum = np.zeros((3,3))
        K_sum = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                sigma_sum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j])
                sigmaSsum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j] \
                        * (self.ek - mu) * constants.elementary_charge)
                K_sum[i,j] = np.sum(dfde_factor * tau * vel_tensor[:,:,i,j] * \
                    ((self.ek - mu) * constants.elementary_charge) **2 )
                
        sigma_sum *= self.weight
        sigmaSsum *= self.weight
        K_sum *= self.weight

        total_volumn = self.nk * self.cell_volume * 1e-30 # m3
        sigma_ij = constants.elementary_charge ** 2 * sigma_sum / total_volumn
        sigmaSij = -1 * constants.elementary_charge * sigmaSsum / total_volumn / temp
        k_ij = K_sum / total_volumn / temp

        Sij = np.linalg.inv(sigma_ij) @ sigmaSij
        # seebeck positive for hole, negative for electron
        return sigma_ij, Sij, k_ij