from math import pi, sqrt, cos, sin
from cmath import phase
import cmath

meson_quark = { 'B0': 'bd', 'Bs': 'bs', 'K0': 'sd', 'D0': 'cu' }

def bag_msbar2rgi(alpha_s, meson):
    """Conversion factor between renormalization group invariant (RGI) defintion
    $\hat B$ of the bag parameter and the $\overline{\mathrm{MS}}$ definition
    $B(mu)$:
    $$\hat B = b_B^{(n_f)} B(mu)$$

    See e.g. eq. (84) in arXiv:1011.4408.
    """
    J={}
    if meson in ['B0', 'Bs']: # nf=5
        J = 5165/3174.
        g = 6/23
    elif meson == 'K0': # nf=3
        J = 307/162.
        g = 2/9.
    return alpha_s**(-g) * (1 + alpha_s/(4*pi) * J)

def DeltaM(M12, G12):
    r"""Meson mixing mass difference $\Delta M$ as a function of $M_{12}$ and
    $\Gamma_{12}$."""
    aM12 = abs(M12)
    aG12 = abs(G12)
    phi12 = phase(-M12/G12)
    return sqrt(4*aM12**2 + aG12**2 + sqrt(16*aM12**4 + 16*aM12**2*aG12**2
                    + aG12**4 + 8*aM12**2*aG12**2*cos(2*phi12)))/sqrt(2)

def DeltaGamma(M12, G12):
    r"""Meson mixing decay width difference $\Delta\Gamma$ as a function of
    $M_{12}$ and $\Gamma_{12}$."""
    aM12 = abs(M12)
    aG12 = abs(G12)
    phi12 = phase(-M12/G12)
    return sqrt(2)*cmath.sqrt(-4*aM12**2 - aG12**2 +
                sqrt(16*aM12**4 + 8*aM12**2*aG12**2 +
                    aG12**4 + 16*aM12**2*aG12**2*cos(phi12)**2)).real

def q_over_p(M12, G12):
    r"""Ratio $q/p$ as a function of $M_{12}$ and $\Gamma_{12}$."""
    DM = DeltaM(M12, G12)
    DG = DeltaGamma(M12, G12)
    return -cmath.sqrt((2*M12.conjugate()-1j*G12.conjugate())/(2*M12-1j*G12))

def a_fs(M12, G12):
    r"""Flavour-specific CP asymmetry in meson mixing as a function of
    $M_{12}$ and $\Gamma_{12}$."""
    aM12 = abs(M12)
    aG12 = abs(G12)
    phi12 = phase(-M12/G12)
    return aG12 / aM12 * sin(phi12)
