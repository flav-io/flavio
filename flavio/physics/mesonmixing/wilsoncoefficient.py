r"""$\Delta F=2$ Wilson coefficient in the Standard Model."""

from math import log,pi,sqrt
from flavio.physics.mesonmixing.common import meson_quark
from flavio.physics import ckm
import flavio

def F_box(x, y):
    r"""$\Delta F=2$ box loop function.

    Parameters
    ----------

    - `x`:
        $m_{u_i}^2/m_W^2$
    - `y`:
        $m_{u_j}^2/m_W^2$
    """
    if x == y:
        return ((4+4 * x-15 * x**2+x**3)/(4 * (-1+x)**2)
              +(x * (-4+4 * x+3 * x**2) * log(x))/(2 * (-1+x)**3))
    return ((4-7 * x * y)/(4-4 * x-4 * y+4 * x * y)
        +(x**2 * (4+(-8+x) * y) * log(x))/(4 * (-1+x)**2 * (x-y))
        -((4+x * (-8+y)) * y**2 * log(y))/(4 * (x-y) * (-1+y)**2))

def S0_box(x, y, xu=0):
    r"""$\Delta F=2$ box loop function $S_0(x, y, x_u)$."""
    return F_box(x, y) + F_box(xu, xu) - F_box(x, xu) - F_box(y, xu)

def df2_prefactor(par):
    GF = par['GF']
    mW = par['m_W']
    return -GF**2/(4.*pi**2) * mW**2


def cvll_d(par, meson, scale=80):
    r"""Contributions to the Standard Model Wilson coefficient $C_V^{LL}$
    for $B^0$, $B_s$, and $K$ mixing at the matching scale.

    The Hamiltonian is defined as
    $$\mathcal H_{\mathrm{eff}} = - C_V^{LL} O_V^{LL},$$
    where $O_V^{LL} = (\bar d_L^i\gamma^\mu d_L^j)^2$ with $i<j$.

    Parameters
    ----------

    - `par`:
        parameter dictionary
    - `meson`:
        should be one of `'B0'`, `'Bs'`, or `'K0'`

    Returns
    -------
    a tuple of three complex numbers `(C_tt, C_cc, C_ct)` that contain the top-,
    charm-, and charm-top-contribution to the Wilson coefficient. This
    separation is necessary as they run differently.
    """
    mt = flavio.physics.running.running.get_mt(par, scale)
    mc = flavio.physics.running.running.get_mc(par, scale)
    mu = flavio.physics.running.running.get_mu(par, scale)
    mW = par['m_W']
    xt = mt**2/mW**2
    xc = mc**2/mW**2
    xu = mu**2/mW**2
    di_dj = meson_quark[meson]
    xi_t = ckm.xi('t',di_dj)(par)
    xi_c = ckm.xi('c',di_dj)(par)
    N = df2_prefactor(par)
    C_cc = N * xi_c**2     * S0_box(xc, xc, xu)
    C_tt = N * xi_t**2     * S0_box(xt, xt, xu)
    C_ct = N * 2*xi_c*xi_t * S0_box(xc, xt, xu)
    return (C_tt, C_cc, C_ct)
