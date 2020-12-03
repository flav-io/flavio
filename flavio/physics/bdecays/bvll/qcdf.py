r"""QCD factorization corrections to $B\to V\ell^+\ell^-$ at low $q^2$"""


from math import pi,exp
from cmath import sqrt,atan,log
import numpy as np
from scipy.special import eval_gegenbauer
from flavio.physics import ckm
from flavio.math.functions import li2, ei
from flavio.physics.running import running
from flavio.physics.bdecays.common import meson_spectator, quark_charge, meson_quark
from flavio.physics.bdecays import matrixelements
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K, beta_l
import flavio
import warnings


# B->V, LO weak annihaltion

# auxiliary function to get the input needed all the time
def get_input(par, B, V, scale):
    mB = par['m_'+B]
    mb = running.get_mb_pole(par)
    mc = running.get_mc_pole(par)
    alpha_s = running.get_alpha(par, scale)['alpha_s']
    q = meson_spectator[(B,V)] # spectator quark flavour
    qiqj = meson_quark[(B,V)]
    eq = quark_charge[q] # charge of the spectator quark
    ed = -1/3.
    eu = 2/3.
    xi_t = ckm.xi('t', qiqj)(par)
    xi_u = ckm.xi('u', qiqj)(par)
    eps_u = xi_u/xi_t
    return mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj

# see eqs. (18), (50), (79) of hep-ph/0106067v2
# and eqs. (47), (48) of hep-ph/0412400
def Cq34(q2, par, wc, B, V, scale):
    flavio.citations.register("Beneke:2001at")
    flavio.citations.register("Beneke:2004dp")
    # this is -C_q^12 (for q=u) or C_q^34 - eps_u * C_q^12 (for q=d,s) of hep-ph/0412400
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    T_t = wc['C3_'+qiqj] + 4/3.*(wc['C4_'+qiqj] + 12*wc['C5_'+qiqj] + 16*wc['C6_'+qiqj])
    # the (u) contribution depends on the flavour of the spectator quark:
    if q == 'u':
        T_u = -3*(wc['C2_'+qiqj])
    elif q == 'd' or q == 's':
        if V == 'omega':
            T_u = -(4/3. * wc['C1_'+qiqj] + wc['C2_'+qiqj])
            # there is also an additional contribution to T_t
            T_t = T_t + 6 * 2 * (wc['C3_'+qiqj] + 10*wc['C5_'+qiqj])
        elif V == 'rho0':
            T_u = +(4/3. * wc['C1_'+qiqj] + wc['C2_'+qiqj])
        elif V == 'K*0':
            T_u = 0
        elif V == 'phi':
            T_u = 0
            # there is also an additional contribution to T_t
            T_t = T_t + 6 * (wc['C3_'+qiqj] + 10*wc['C5_'+qiqj])
    return T_t + eps_u * T_u

def T_para_minus_WA(q2, par, wc, B, V, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    return -eq * 4*mB/mb * Cq34(q2, par, wc, B, V, scale)


# B->V, weak annihilation at O(1/mb)

# (51) of hep-ph/0412400

def T_perp_WA_PowC_1(q2, par, wc, B, V, scale):
    flavio.citations.register("Beneke:2004dp")
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    # NB, the remaining prefactors are added below in the function T_perp
    return -eq * 4/mb * (wc['C3_'+qiqj] + 4/3.*(wc['C4_'+qiqj] + 3*wc['C5_'+qiqj] + 4*wc['C6_'+qiqj]))

def T_perp_WA_PowC_2(q2, par, wc, B, V, scale):
    flavio.citations.register("Beneke:2004dp")
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    # NB, the remaining prefactors are added below in the function T_perp
    return eq * 2/mb * Cq34(q2, par, wc, B, V, scale)


# B->V, NLO spectator scattering

# (23)-(26) of hep-ph/0106067v2

# chromomagnetic dipole contribution
def T_perp_plus_O8(q2, par, wc, B, V, u, scale):
    flavio.citations.register("Beneke:2001at")
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    ubar = 1 - u
    return - (alpha_s/(3*pi)) * 4*ed*wc['C8eff_'+qiqj]/(u + ubar*q2/mB**2)

def T_para_minus_O8(q2, par, wc, B, V, u, scale):
    flavio.citations.register("Beneke:2001at")
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    ubar = 1 - u
    return (alpha_s/(3*pi)) * eq * 8 * wc['C8eff_'+qiqj]/(ubar + u*q2/mB**2)


# 4-quark operator contribution
def T_perp_plus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    ubar = 1 - u
    t_mc = t_perp(q2=q2, u=u, mq=mc, par=par, B=B, V=V)
    t_mb = t_perp(q2=q2, u=u, mq=mb, par=par, B=B, V=V)
    t_0  = t_perp(q2=q2, u=u, mq=0,  par=par, B=B, V=V)
    T_t = (alpha_s/(3*pi)) * mB/(2*mb)*(
          eu * t_mc * (-wc['C1_'+qiqj]/6. + wc['C2_'+qiqj] + 6*wc['C6_'+qiqj])
        + ed * t_mb * (wc['C3_'+qiqj] - wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] + (10.*wc['C6_'+qiqj])/3.
                        + mb/mB*(-wc['C3_'+qiqj] + wc['C4_'+qiqj]/6 - 4 * wc['C5_'+qiqj] + (2 * wc['C6_'+qiqj])/3))
        + ed * t_0  * (wc['C3_'+qiqj] - wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] - 8*wc['C6_'+qiqj]/3.)
        )
    T_u = ( (alpha_s/(3*pi)) * eu * mB/(2*mb) * ( t_mc - t_0 )
                                * ( wc['C2_'+qiqj] - wc['C1_'+qiqj]/6.) )
    return T_t + eps_u * T_u

def T_para_plus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    ubar = 1 - u
    t_mc = t_para(q2=q2, u=u, mq=mc, par=par, B=B, V=V)
    t_mb = t_para(q2=q2, u=u, mq=mb, par=par, B=B, V=V)
    t_0  = t_para(q2=q2, u=u, mq=0,  par=par, B=B, V=V)
    T_t = (alpha_s/(3*pi)) * mB/mb*(
          eu * t_mc * (-wc['C1_'+qiqj]/6. + wc['C2_'+qiqj] + 6*wc['C6_'+qiqj])
        + ed * t_mb * (wc['C3_'+qiqj] - wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] + 10*wc['C6_'+qiqj]/3.)
        + ed * t_0 * (wc['C3_'+qiqj] - wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] - 8*wc['C6_'+qiqj]/3.)
        )
    T_u = ( (alpha_s/(3*pi)) * eu * mB/mb * ( t_mc - t_0 )
                                * ( wc['C2_'+qiqj] - wc['C1_'+qiqj]/6.) )
    return T_t + eps_u * T_u

def T_para_minus_QSS(q2, par, wc, B, V, u, scale):
    mB, mb, mc, alpha_s, q, eq, ed, eu, eps_u, qiqj = get_input(par, B, V, scale)
    ubar = 1 - u
    h_mc = matrixelements.h(ubar*mB**2 + u*q2, mc, scale)
    h_mb = matrixelements.h(ubar*mB**2 + u*q2, mb, scale)
    h_0  = matrixelements.h(ubar*mB**2 + u*q2, 0, scale)
    T_t =  (alpha_s/(3*pi)) * eq * 6 * mB/mb*(
          h_mc * (-wc['C1_'+qiqj]/6. + wc['C2_'+qiqj] + wc['C4_'+qiqj] + 10*wc['C6_'+qiqj])
        + h_mb * (wc['C3_'+qiqj] + 5*wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] + 22*wc['C6_'+qiqj]/3.)
        + h_0 * (wc['C3_'+qiqj] + 17*wc['C4_'+qiqj]/6. + 16*wc['C5_'+qiqj] + 82*wc['C6_'+qiqj]/3.)
        - 8/27. * (-15*wc['C4_'+qiqj]/2. + 12*wc['C5_'+qiqj] - 32*wc['C6_'+qiqj])
        )
    T_u = ( (alpha_s/(3*pi)) * eq * 6*mB/mb * ( h_mc - h_0 )
                                * ( wc['C2_'+qiqj] - wc['C1_'+qiqj]/6.) )
    return T_t + eps_u * T_u

def En_V(mB, mV, q2):
    """Energy of the vector meson"""
    return (mB**2 - q2 + mV**2)/(2*mB)

# (27) of hep-ph/0106067v2
def t_perp(q2, u, mq, par, B, V):
    flavio.citations.register("Beneke:2001at")
    mB = par['m_'+B]
    mV = par['m_'+V]
    EV = En_V(mB, mV, q2)
    ubar = 1 - u
    if q2 == 0.: # limiting case for q2->0: eq. (33) of hep-ph/0106067v2
        x0 = sqrt(1/4. - mq**2/(ubar * mB**2))
        xp = 1/2. + x0
        xm = 1/2. - x0
        return 4/ubar * (1 + 2*mq**2/(ubar*mB**2) * (L1(xp) + L1(xm)) )
    return ((2*mB)/(ubar * EV) * i1_bfs(q2, u, mq, mB)
            + q2/(ubar**2 * EV**2) * B0diffBFS(q2, u, mq, mB))

# (28) of hep-ph/0106067v2
def t_para(q2, u, mq, par, B, V):
    flavio.citations.register("Beneke:2001at")
    mB = par['m_'+B]
    mV = par['m_'+V]
    EV = En_V(mB, mV, q2)
    ubar = 1 - u
    return ((2*mB)/(ubar * EV) * i1_bfs(q2, u, mq, mB)
            + (ubar*mB**2 + u*q2)/(ubar**2 * EV**2) * B0diffBFS(q2, u, mq, mB))

def B0diffBFS(q2, u, mq, mB):
    ubar = 1 - u
    if mq == 0.:
        return -log(-(2/q2)) + log(-(2/(q2*u + mB**2 * ubar)))
    return B0(ubar * mB**2 + u * q2, mq) - B0(q2, mq)

# (29) of hep-ph/0106067v2
def B0(s, mq):
    flavio.citations.register("Beneke:2001at")
    if s==0.:
        return -2.
    if 4*mq**2/s == 1.:
        return 0.
    # to select the right branch of the complex arctangent, need to
    # interpret m^2 as m^2-i\epsilon
    iepsilon = 1e-8j
    return -2*sqrt(4*(mq**2-iepsilon)/s - 1) * atan(1/sqrt(4*(mq**2-iepsilon)/s - 1))

# (30), (31) of hep-ph/0106067v2
def i1_bfs(q2, u, mq, mB):
    flavio.citations.register("Beneke:2001at")
    ubar = 1 - u
    iepsilon = 1e-8j
    mq2 = mq**2 - iepsilon
    x0 = sqrt(1/4. - mq2/(ubar * mB**2 + u * q2))
    xp = 1/2. + x0
    xm = 1/2. - x0
    y0 = sqrt(1/4. - mq2/q2)
    yp = 1/2. + y0
    ym = 1/2. - y0
    return 1 + (2 * mq2)/(ubar * (mB**2 - q2)) * (L1(xp) + L1(xm) - L1(yp) - L1(ym))

# (32) of hep-ph/0106067v2
def L1(x):
    flavio.citations.register("Beneke:2001at")
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
    flavio.citations.register("Beneke:2001at")
    mB = par['m_'+B]
    mb = running.get_mb_pole(par)
    LambdaBar = mB - mb
    w0 = 2*LambdaBar/3
    return 1/(exp(-q2/mB/w0)/w0 * (-ei(q2/mB/w0) + 1j*pi))
def lB_plus(par, B):
    flavio.citations.register("Beneke:2001at")
    mB = par['m_'+B]
    mb = running.get_mb_pole(par)
    LambdaBar = mB - mb
    return 2*LambdaBar/3


# (15) of hep-ph/0106067v2

def T_para(q2, par, wc, B, V, scale,
           include_WA=True, include_O8=True, include_QSS=True):
    if not include_WA and not include_O8 and not include_QSS:
        raise ValueError("At least one contribution to the QCDF corrections has to be switched on")
    mB = par['m_'+B]
    mV = par['m_'+V]
    mc = running.get_mc_pole(par)
    fB = par['f_'+B]
    fVpara = par['f_' + V]
    EV = En_V(mB, mV, q2)
    N = pi**2 / 3. * fB * fVpara / mB * (mV/EV)
    a1_para = par['a1_para_'+V]
    a2_para = par['a2_para_'+V]
    def phiV_para(u):
        return phiV(u, a1_para, a2_para)
    def T_minus(u):
        T = 0
        if include_WA:
            T += T_para_minus_WA(q2=q2, par=par, wc=wc, B=B, V=V, scale=scale)
        if include_O8:
            T += T_para_minus_O8(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale)
        if include_QSS:
            T += T_para_minus_QSS(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale)
        return N / lB_minus(q2=q2, par=par, B=B) * phiV_para(u) * T
    def T_plus(u):
        if include_QSS:
            return N / lB_plus(par=par, B=B) * phiV_para(u) * (
                T_para_plus_QSS(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale))
        else:
            return 0
    u_sing = (mB**2 - 4*mc**2)/(-q2 + mB**2)
    if u_sing < 1:
        points = [u_sing]
    else:
        points = None
    T_tot = flavio.math.integrate.nintegrate_complex( lambda u: T_plus(u) + T_minus(u), 0, 1, points=points)
    return T_tot



def T_perp(q2, par, wc, B, V, scale,
           include_WA=True, include_O8=True, include_QSS=True):
    if not include_WA and not include_O8 and not include_QSS:
        raise ValueError("At least one contribution to the QCDF corrections has to be switched on")
    mB = par['m_'+B]
    mV = par['m_'+V]
    mc = running.get_mc_pole(par)
    EV = En_V(mB, mV, q2)
    fB = par['f_'+B]
    fVperp = flavio.physics.running.running.get_f_perp(par, V, scale)
    fVpara = par['f_'+V]
    N = pi**2 / 3. * fB * fVperp / mB
    a1_perp = par['a1_perp_'+V]
    a2_perp = par['a2_perp_'+V]
    def phiV_perp(u):
        return phiV(u, a1_perp, a2_perp)
    def T_minus(u):
        return 0
    def T_plus(u):
        T = 0
        if include_O8:
            T += T_perp_plus_O8(q2=q2, par=par, wc=wc, B=B, V=V, u=u, scale=scale)
        if include_QSS:
            T += T_perp_plus_QSS(q2, par, wc, B, V, u, scale)
        return N / lB_plus(par=par, B=B) * phiV_perp(u) * T
    def T_powercorr_1(u):
        T=0
        if include_WA:
            T += T_perp_WA_PowC_1(q2, par, wc, B, V, scale)
        ubar = 1 - u
        # cf. (51) of hep-ph/0412400
        return N * phiV_perp(u)/(ubar+u*q2/mB**2) * T
    u_sing = (mB**2 - 4*mc**2)/(-q2 + mB**2)
    if u_sing < 1:
        points = [u_sing]
    else:
        points = None
    T_tot = flavio.math.integrate.nintegrate_complex( lambda u: T_plus(u) + T_minus(u) + T_powercorr_1(u), 0, 1, points=points)
    if include_WA:
    # cf. (51) of hep-ph/0412400
        T_tot += N / lB_plus(par=par, B=B) * fVpara/fVperp * mV/(1-q2/mB**2) * T_perp_WA_PowC_2(q2, par, wc, B, V, scale)
    return T_tot


def transversity_amps_qcdf(q2, wc, par, B, V, **kwargs):
    """QCD factorization corrections to B->Vll transversity amplitudes."""
    mB = par['m_'+B]
    mV = par['m_'+V]
    scale = config['renormalization scale']['bvll']
    # using the b quark pole mass here!
    mb = running.get_mb_pole(par)
    N = flavio.physics.bdecays.bvll.amplitudes.prefactor(q2, par, B, V)/4
    T_perp_ = T_perp(q2, par, wc, B, V, scale, **kwargs)
    T_para_ = T_para(q2, par, wc, B, V, scale, **kwargs)
    ta = {}
    ta['perp_L'] = N * sqrt(2)*2 * (mB**2-q2) * mb / q2 * T_perp_
    ta['perp_R'] =  ta['perp_L']
    ta['para_L'] = -ta['perp_L']
    ta['para_R'] =  ta['para_L']
    ta['0_L'] = ( N * mb * (mB**2 - q2)**2 )/(mB**2 * mV * sqrt(q2)) * T_para_
    ta['0_R'] = ta['0_L']
    ta['t'] = 0
    ta['S'] = 0
    return ta

def helicity_amps_qcdf(q2, wc, par, B, V, **kwargs):
    if q2 > 6:
        warnings.warn("The QCDF corrections should not be trusted for q2 above 6 GeV^2")
    mB = par['m_'+B]
    mV = par['m_'+V]
    X = sqrt(lambda_K(mB**2,q2,mV**2))/2.
    ta = transversity_amps_qcdf(q2, wc, par, B, V, **kwargs)
    h = flavio.physics.bdecays.angular.transversity_to_helicity(ta)
    return h
