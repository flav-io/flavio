from math import pi,exp
from cmath import sqrt,atan,log
import mpmath
import numpy as np
from scipy.special import eval_gegenbauer
from scipy.integrate import quad
from flavio.physics import ckm
from flavio.physics.functions import li2, ei
from flavio.physics.running import running
from flavio.physics.bdecays.common import meson_spectator, quark_charge, meson_quark
from flavio.physics.bdecays import matrixelements

def complex_quad(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

# B->V, LO weak annihaltion

# auxiliary function to get the input needed all the time
def get_input(par, B, V, scale):
    mB = par[('mass',B)]
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    q = meson_spectator[(B,V)] # spectator quark flavour
    eq = quark_charge[q] # charge of the spectator quark
    ed = -1/3.
    eu = 2/3.
    xi_t = ckm.xi('t', meson_quark[(B,V)])(par)
    xi_u = ckm.xi('u', meson_quark[(B,V)])(par)
    eps_u = xi_u/xi_t
    return mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u

# see eqs. (18), (19), (50), (79) of hep-ph/0106067v2
def T_para_minus_WA(q2, par, wc, B, V, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    if q == 'u':
        # for an up-type spectator quark, additional contribution from current-current operators
        xi_t = ckm.xi('t', meson_quark[(B,V)])
        xi_u = ckm.xi('u', meson_quark[(B,V)])
        T_cc = eq * 4*mB/mb * 3*wc['C2'] * xi_u/xi_t
    # QCD penguin contribution to LO WA
    T_p = -eq * 4*mB/mb * (wc['C3'] + 4/3.*(wc['C4'] + 12*wc['C5'] + 16*wc['C6']))
    return T_p


# B->V, NLO spectator scattering

# (23)-(26) of of hep-ph/0106067v2

# chromomagnetic dipole contribution
def T_perp_plus_O8(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    ubar = 1 - u
    return - (alpha_s/(3*pi)) * 4*ed*wc['C8eff']/(u + ubar*q2/mB**2)

def T_para_minus_O8(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    ubar = 1 - u
    return (alpha_s/(3*pi)) * eq * 8 * wc['C8eff']/(ubar + u*q2/mB**2)


# 4-quark operator contribution
def T_perp_plus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    ubar = 1 - u
    t_mc = t_perp(q2=q2, u=u, mq=mc, par=par, B=B, V=V)
    t_mb = t_perp(q2=q2, u=u, mq=mb, par=par, B=B, V=V)
    t_0  = t_perp(q2=q2, u=u, mq=0,  par=par, B=B, V=V)
    T_t = (alpha_s/(3*pi)) * mB/(2*mb)*(
          eu * t_mc * (-wc['C1']/6. + wc['C2'] + 6*wc['C6'])
        + ed * t_mb * (wc['C3'] - wc['C4']/6. + 16*wc['C5'] + (10.*wc['C6'])/3.
                        + mb/mB*(-wc['C3'] + wc['C4']/6 - 4 * wc['C5'] + (2 * wc['C6'])/3))
        + ed * t_0  * (-wc['C3'] + wc['C4']/6. - 16*wc['C5'] + 8*wc['C6']/3.)
        )
    T_u = ( (alpha_s/(3*pi)) * eu * mB/(2*mb) * ( t_mc - t_0 )
                                * ( wc['C2'] - wc['C1']/6.) )
    return T_t + eps_u * T_u

def T_para_plus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    ubar = 1 - u
    t_mc = t_para(q2=q2, u=u, mq=mc, par=par, B=B, V=V)
    t_mb = t_para(q2=q2, u=u, mq=mb, par=par, B=B, V=V)
    t_0  = t_para(q2=q2, u=u, mq=0,  par=par, B=B, V=V)
    T_t = (alpha_s/(3*pi)) * mB/mb*(
          eu * t_mc * (-wc['C1']/6. + wc['C2'] + 6*wc['C6'])
        + ed * t_mb * (wc['C3'] - wc['C4']/6. + 16*wc['C5'] + 10*wc['C6']/3.)
        + ed * t_0 * (-wc['C3'] + wc['C4']/6. - 16*wc['C5'] + 8*wc['C6']/3.)
        )
    T_u = ( (alpha_s/(3*pi)) * eu * mB/mb * ( t_mc - t_0 )
                                * ( wc['C2'] - wc['C1']/6.) )
    return T_t + eps_u * T_u

def T_para_minus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u = get_input(par, B, V, scale)
    ubar = 1 - u
    h_mc = matrixelements.h(ubar*mB**2 + u*q2, mc, scale)
    h_mb = matrixelements.h(ubar*mB**2 + u*q2, mb, scale)
    h_0  = matrixelements.h(ubar*mB**2 + u*q2, 0, scale)
    T_t =  (alpha_s/(3*pi)) * eq * 6 * mB/mb*(
          h_mc * (-wc['C1']/6. + wc['C2'] + wc['C4'] + 10*wc['C6'])
        + h_mb * (wc['C3'] + 5*wc['C4']/6. + 16*wc['C5'] + 22*wc['C6']/3.)
        + h_0 * (wc['C3'] + 17*wc['C4']/6. + 16*wc['C5'] + 82*wc['C6']/3.)
        - 8/27. * (-15*wc['C4']/2. + 12*wc['C5'] - 32*wc['C6'])
        )
    T_u = ( (alpha_s/(3*pi)) * eq * 6*mB/mb * ( h_mc - h_0 )
                                * ( wc['C2'] - wc['C1']/6.) )
    return T_t + eps_u * T_u

def En_V(mB, mV, q2):
    """Energy of the vector meson"""
    return (mB**2 - q2 + mV**2)/(2*mB)

# (27) of of hep-ph/0106067v2
def t_perp(q2, u, mq, par, B, V):
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    EV = En_V(mB, mV, q2)
    ubar = 1 - u
    return ((2*mB)/(ubar * EV) * i1_bfs(q2, u, mq, mB)
            + q2/(ubar**2 * EV**2) * B0diffBFS(q2, u, mq, mB))

# (28) of of hep-ph/0106067v2
def t_para(q2, u, mq, par, B, V):
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    EV = En_V(mB, mV, q2)
    ubar = 1 - u
    return ((2*mB)/(ubar * EV) * i1_bfs(q2, u, mq, mB)
            + (ubar*mB + u*q2)/(ubar**2 * EV**2) * B0diffBFS(q2, u, mq, mB))

def B0diffBFS(q2, u, mq, mB):
    ubar = 1 - u
    if mq == 0.:
        return -log(-(2/q2)) + log(-(2/(q2*u + mB**2 * ubar)))
    return B0(ubar * mB**2 + u * q2, mq) - B0(q2, mq)

# (29) of of hep-ph/0106067v2
def B0(s, mq):
    if s==0.:
        return -2.
    if 4*mq**2/s == 1.:
        return 0.
    # to select the right branch of the complex arctangent, need to
    # interpret m^2 as m^2-i\epsilon
    iepsilon = 1e-8j
    return -2*sqrt(4*mq**2/s - 1) * atan(1/sqrt(4*(mq**2-iepsilon)/s - 1))

# (30), (31) of of hep-ph/0106067v2
def i1_bfs(q2, u, mq, mB):
    ubar = 1 - u
    x0 = sqrt(1/4. - mq**2/(ubar * mB**2 + u * q2))
    xp = 1/2. + x0
    xm = 1/2. - x0
    y0 = sqrt(1/4. - mq**2/q2)
    yp = 1/2. + y0
    ym = 1/2. - y0
    return 1 + (2 * mq**2)/(ubar * (mB**2 - q2)) * (L1(xp) + L1(xm) - L1(yp) - L1(ym))

# (32) of of hep-ph/0106067v2
def L1(x):
    if x == 0.:
        return -(pi**2/6.)
    elif x == 1.:
        return 0
    return log((x - 1)/x) * log(1 - x) - pi**2/6. + li2(x/(x - 1))


def phiV(u, a1, a2):
    """Vector meson light-cone distribution amplitude to second order
    in the Gegenbauer expansion."""
    c1 = eval_gegenbauer(1, 3/2., 2*u-1)
    c2 = eval_gegenbauer(2, 3/2., 2*u-1)
    return 6*u * (1-u) * (1 + a1 * c1 + a2 * c2)


# moments of the B meson light-cone distribution amplitude as in
# eq. (54) and surroundings of hep-ph/0106067v2
def lB_minus(q2, par, B):
    mB = par[('mass',B)]
    mb = running.get_mb_pole(par)
    LambdaBar = mB - mb
    w0 = 2*LambdaBar/3
    return 1/(exp(-q2/mB/w0)/w0 * (-ei(q2/mB/w0) + 1j*pi))
def lB_plus(par, B):
    mB = par[('mass',B)]
    mb = running.get_mb_pole(par)
    LambdaBar = mB - mb
    return 2*LambdaBar/3


# (15) of hep-ph/0106067v2

def T_para(q2, par, wc, B, V, scale):
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    mc = running.get_mc_pole(par)
    fB = par[('f',B)]
    fVpara = par[('f_para',V)]
    EV = En_V(mB, mV, q2)
    N = pi**2 / 3. * fB * fVpara / mB * (mV/EV)
    a1_para = par[('a1_para',V)]
    a2_para = par[('a2_para',V)]
    def phiV_para(u):
        return phiV(u, a1_para, a2_para)
    def T_minus(u):
        return N / lB_minus(q2=q2, par=par, B=B) * phiV_para(u) * (
                T_para_minus_WA(q2=q2, par=par, wc=wc, B=B, V=V, scale=scale)
              + T_para_minus_O8(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale)
              + T_para_minus_QSS(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale))
    def T_plus(u):
        return N / lB_plus(par=par, B=B) * phiV_para(u) * (
                T_para_plus_QSS(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale))
    u_sing = (mB**2 - 4*mc**2)/(-q2 + mB**2)
    if u_sing < 1:
        points = [u_sing]
    else:
        points = None
    T_tot = complex_quad( lambda u: T_plus(u) + T_minus(u), 0, 1, points=points, epsrel=0.01, epsabs=0 )[0]
    return T_tot



def T_perp(q2, par, wc, B, V, scale):
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    mc = running.get_mc_pole(par)
    EV = En_V(mB, mV, q2)
    fB = par[('f',B)]
    fVperp = par[('f_perp',V)]
    N = pi**2 / 3. * fB * fVperp / mB
    a1_perp = par[('a1_perp',V)]
    a2_perp = par[('a2_perp',V)]
    def phiV_perp(u):
        return phiV(u, a1_perp, a2_perp)
    def T_minus(u):
        return 0
    def T_plus(u):
        return N / lB_plus(par=par, B=B) * phiV_perp(u) * (
                T_perp_plus_O8(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale)
              + T_perp_plus_QSS(q2, par, wc, B, V, u, scale))
    u_sing = (mB**2 - 4*mc**2)/(-q2 + mB**2)
    if u_sing < 1:
        points = [u_sing]
    else:
        points = None
    T_tot = complex_quad( lambda u: T_plus(u) + T_minus(u), 0, 1, points=points, epsrel=0.01, epsabs=0 )[0]
    return T_tot
