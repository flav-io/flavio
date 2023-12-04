r"""$\Delta F=2$ Wilson coefficient in the Standard Model."""

from math import log,pi,sqrt
from flavio.physics.mesonmixing.common import meson_quark
from flavio.physics import ckm
import flavio
from flavio.config import config

def F_box(x, y):
    r"""$\Delta F=2$ box loop function.

    Parameters
    ----------

    - `x`:
        $m_{u_i}^2/m_W^2$
    - `y`:
        $m_{u_j}^2/m_W^2$
    """
    if x == 0 and y == 0:
        return 1
    elif x == 0:
        return (1 - y + y*log(y))/(-1 + y)**2
    elif y == 0:
        return (1 - x + x*log(x))/(-1 + x)**2
    elif x == y:
        return ((4+4 * x-15 * x**2+x**3)/(4 * (-1+x)**2)
              +(x * (-4+4 * x+3 * x**2) * log(x))/(2 * (-1+x)**3))
    return ((4-7 * x * y)/(4-4 * x-4 * y+4 * x * y)
        +(x**2 * (4+(-8+x) * y) * log(x))/(4 * (-1+x)**2 * (x-y))
        -((4+x * (-8+y)) * y**2 * log(y))/(4 * (x-y) * (-1+y)**2))

def S0_box(x, y, xu=0):
    r"""$\Delta F=2$ box loop function $S_0(x, y, x_u)$."""
    flavio.citations.register("Inami:1980fz")
    return F_box(x, y) + F_box(xu, xu) - F_box(x, xu) - F_box(y, xu)

def cvll_d(par, meson):
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
    C_V^{LL} RI Wilson coefficient including QCD correction factors
    """
    mt = flavio.physics.running.running.get_mt_mt(par)
    mc = par["m_c"]
    mu = par["m_u"]
    mW = par['m_W']
    xt = mt**2/mW**2
    xc = mc**2/mW**2
    xu = mu**2/mW**2
    di_dj = meson_quark[meson]
    GF = par['GF']
    N = -GF**2/(4.*pi**2) * mW**2
    # charm contribution only needed for K mixing! Negligible for B and Bs.
    if meson == 'K0':
        if config['implementation']['K mixing unitarity'] == 'ut':
            # use u-t unitarity (1911.06822)
            xi_t = ckm.xi('t',di_dj)(par)
            xi_u = ckm.xi('u',di_dj)(par)
            C_tt =  xi_t**2    * S0_box(xt, xt, xc)
            C_ut = 2*xi_u*xi_t * S0_box(xu, xt, xc)
            C_uu =  xi_u**2    * S0_box(xu, xu, xc) # Im(C_uu) = 0
            eta_tt = par['eta_tt_'+meson+'_ut']
            eta_ut = par['eta_ut_'+meson+'_ut']
            eta_uu = par['eta_uu_'+meson+'_ut']
            return N * (eta_tt*C_tt + eta_ut*C_ut + eta_uu*C_uu)
        elif config['implementation']['K mixing unitarity'] == 'ct':
            # use traditional c-t unitarity
            xi_t = ckm.xi('t',di_dj)(par)
            xi_c = ckm.xi('c',di_dj)(par)
            C_tt =  xi_t**2    * S0_box(xt, xt, xu)
            C_ct = 2*xi_c*xi_t * S0_box(xc, xt, xu)
            C_cc =  xi_c**2    * S0_box(xc, xc, xu)
            eta_tt = par['eta_tt_'+meson]
            eta_ct = par['eta_ct_'+meson]
            eta_cc = par['eta_cc_'+meson]
            return N * (eta_tt*C_tt + eta_ct*C_ct + eta_cc*C_cc)
        else:
            raise ValueError("Unknown value for K mixing unitarity: {}".format(config['implementation']['K mixing unitarity']))
    else:
        # use traditional c-t unitarity
        xi_t = ckm.xi('t',di_dj)(par)
        C_tt = xi_t**2     * S0_box(xt, xt, xu)
        eta_tt = par['eta_tt_'+meson]
        return N * eta_tt * C_tt
