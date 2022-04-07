import numpy as np

def fact2(n):
    """
    fact2(n) - n!!, double factorial of n
    >>> fact2(8)    384
    >>> fact2(-1)   1
    """
    return np.prod(range(n,0,-2), dtype = float)

powers = {
    "s"      : (0,0,0),
    "px"     : (1,0,0),
    "py"     : (0,1,0),
    "pz"     : (0,0,1),
    "dxy"    : (1,1,0),
    "dyz"    : (0,1,1),
    "dxz"    : (1,0,1),
    "dz2"    : (0,0,2),
    "dx2"    : (2,0,0),
    "dy2"    : (0,2,0)
}
coefficients = {
    "s"      : [(1.0, "s")],
    "px"     : [(1.0, "px")],
    "py"     : [(1.0, "py")],
    "pz"     : [(1.0, "pz")],
    "dxy"    : [(1.0, "dxy")],
    "dyz"    : [(1.0, "dyz")],
    "dxz"    : [(1.0, "dxz")],
    "dz2"    : [(np.sqrt(2)/2, "dz2"),(-0.5, "dx2"),(-0.5, "dy2")],  # 3z2 - r2
    "dx2-y2" : [(np.sqrt(2)/2, "dx2"), (-np.sqrt(2)/2, "dy2")],      # x2 - y2
}

def wavefunction(n, name, origin, xs, ys, zs):
    mesh = np.stack([xs,ys,zs], axis = -1)
    result = np.zeros_like(xs)
    for coeff, na in coefficients[name]:
        bf = pgbf(n, powers[na], origin)
        result += coeff * bf(mesh)
    return result  

def molecularfuntion(origins, mo_coeff, xs, ys, zs):
    """
    mo_coeff is a list giving the coefficients in the given order, 
    s, px, py, pz, dxy, dyz, dxz, dz2, dx2-y2
    n = 1 by default
    """
    default_n = 1
    possible_ao = "s px py pz dxy dyz dxz dz2 dx2-y2".split()
    origins = np.array(origins, dtype = float)
    mo_coeff = np.array(mo_coeff, dtype = float)
    natom, _ = origins.shape
    natom2, ncoeff = mo_coeff.shape 
    assert natom == natom2
    assert ncoeff <= 9

    mesh = np.stack([xs,ys,zs], axis = -1)
    result = np.zeros_like(xs)

    for pos, aos in zip(origins, mo_coeff):
        for i, c in enumerate(aos):
            if np.abs(c) < 1e-3: continue
            tmp = np.zeros_like(xs)
            for coeff, na in coefficients[possible_ao[i]]:
                bf = pgbf(default_n, powers[na], pos)
                tmp += coeff * bf(mesh)
            result += c * tmp
    return result


class pgbf():
    """
    callable
    Construct a primitive gaussian basis functions, as in 
    Hermann, Gunter, Vincent Pohl, Jean Christophe Tremblay, Beate Paulus, Hans-Christian Hege, and Axel Schild. 2016. “ORBKIT: A Modular Python Toolbox for Cross-Platform Postprocessing of Quantum Chemical Wavefunction Data.” Journal of Computational Chemistry 37 (16): 1511–20. https://doi.org/10.1002/jcc.24358.
    Eq. 3) - 4)
    """
    contracted = False
    def __init__(self, alpha: int, lmn:tuple = (0,0,0), origin=(0,0,0)):
        self.norm = 1
        assert len(origin) == 3
        assert len(lmn) == 3

        self.origin = np.array(origin,dtype = 'float')
        self.alpha = float(alpha)
        self.lmn = lmn
        self.N = self._normalize(self.alpha, *self.lmn)

    def __call__(self,xyzs):
        "Compute the amplitude of the PGBF at point x,y,z"
        l,m,n = self.lmn
        r = np.array(xyzs, dtype = float)
        assert r.shape[-1] == 3

        delta = r - self.origin
        r2 = np.linalg.norm(delta, axis=-1)
        return self.N * (delta[...,0]**l) * (delta[...,1]**m) * (delta[...,2]**n) * np.exp(- self.alpha * r2)

    def _normalize(self, alpha, l, m, n):
        "Normalize basis function. From THO eq. 2.2"
        return np.sqrt( pow(2,2*(l+m+n)+1.5) * pow(alpha,l+m+n+1.5) /
                        fact2(2*l-1)/fact2(2*m-1)/
                        fact2(2*n-1)/pow(np.pi,1.5)
                    )

if __name__ == "__main__":
    func = pgbf(1, (1,0,0))
    print(func([[1.0,1.0,1.0]]))