from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics.bdecays import matrixelements
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict

"""Functions for exclusive $B\to P\ell^+\ell^-$ decays."""

def prefactor(q2, par, B, P, lep):
    GF = par['Gmu']
    scale = config['bdecays']['scale_bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mP = par[('mass',P)]
    tauB = par[('lifetime',B)]
    la = lambda_K(mB**2,q2,mP**2)
    if la < 0:
        return 0.
    if q2 <= 4*ml**2:
        return 0
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    return ( sqrt((GF**2 * alphaem**2)/(2**9 * pi**5 * mB**3)
            *sqrt(la) * beta_l(ml, q2)) * xi_t )

def amps(q2, wc, par, B, P, lep):
    # setting the parameters and prefactor
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    scale = config['bdecays']['scale_bpll']
    mb = running.get_mb(par, scale)
    mP = par[('mass',P)]
    # functions for the "effective" Wilson coefficients including the matrix
    # elements of 4-quark operators
    #   a) LO Q1-6
    xi_u = ckm.xi('u',meson_quark[(B,P)])(par)
    xi_t = ckm.xi('t',meson_quark[(B,P)])(par)
    Yq2 = matrixelements.Y(q2, wc, par, scale) + (xi_u/xi_t)*matrixelements.Yu(q2, wc, par, scale)
    #   b) NNLO Q1,2
    delta_C7 = matrixelements.delta_C7(par=par, wc=wc, q2=q2, scale=scale, qiqj=meson_quark[(B,P)])
    delta_C9 = matrixelements.delta_C9(par=par, wc=wc, q2=q2, scale=scale, qiqj=meson_quark[(B,P)])
    c7pl = wc['C7eff'] + wc['C7effp'] + delta_C7
    c9pl = wc['C9'] + wc['C9p']       + delta_C9 + Yq2
    c10pl = wc['C10'] + wc['C10p']
    cspl = wc['CS'] + wc['CSp']
    cppl = wc['CP'] + wc['CPp']
    N = prefactor(q2, par, B, P, lep)
    # form factors
    ff = FF.parametrizations['btop_lattice'].get_ff(meson_ff[(B,P)], q2, par)
    # amplitudes
    F = {}
    F['A'] = N * c10pl * ff['f+']
    F['V'] = N * ( c9pl * ff['f+'] + c7pl * ff['fT'] * 2*mb/(mB+mP) )
    F['P'] = (- N * ml * c10pl * (ff['f+'] + (ff['f+']-ff['f0'])*((mB**2 - mP**2)/q2))
               + N * cppl * ff['f0'] * (mB**2 - mP**2)/mb/2.)
    F['S'] = N * cspl * ff['f0'] * (mB**2 - mP**2)/mb/2.
    return F


def amps_bar(q2, wc, par, B, P, lep):
    par_c = conjugate_par(par)
    wc_c = conjugate_wc(wc)
    return amps(q2, wc_c, par_c, B, P, lep)


def angulardist(amps, q2, par, B, P, lep):
    r"""Returns the angular coefficients of the 2-fold differential decay
    distribution of a $B\to P\ell**+\ell**-$ decay as defined in ...
    """
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mP = par[('mass',P)]
    la = lambda_K(mB**2,q2,mP**2)
    F = amps
    ac = {}
    ac['a'] = (la/4*(abs(F['A'])**2 + abs(F['V'])**2)
              + q2*(beta_l(ml,q2)**2 * abs(F['S'])**2 + abs(F['P'])**2)
              + 2*ml*(mB**2 - mP**2 + q2) * (F['P']*F['A'].conj()).real
              + 4*ml**2 * mB**2 * abs(F['A'])**2)
    ac['b'] = 2*ml*sqrt(la) * beta_l(ml,q2) * (F['S']*F['V'].conj()).real
    ac['c'] = (-(la/4) * beta_l(ml,q2)**2) * (abs(F['A'])**2 + abs(F['V'])**2)
    return ac


def bpll_obs(function, q2, wc_obj, par, B, P, lep):
    scale = config['bdecays']['scale_bpll']
    wc = wctot_dict(wc_obj, 'df1_' + meson_quark[(B,P)], scale, par)
    a = amps(q2, wc, par, B, P, lep)
    a_bar = amps_bar(q2, wc, par, B, P, lep)
    J     = angulardist(a,     q2, par, B, P, lep)
    J_bar = angulardist(a_bar, q2, par, B, P, lep)
    return function(J, J_bar)

def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def AFB_num(J):
    return J['b']

def FH_num(J):
    return 2 * (J['a'] + J['c'])

def dGdq2_cpaverage(J, J_bar):
    return (dGdq2(J) + dGdq2(J_bar))/2.

def AFB_cpaverage_num(J, J_bar):
    return (AFB_num(J) + AFB_num(J_bar))/2.

def FH_cpaverage_num(J, J_bar):
    return (FH_num(J) + FH_num(J_bar))/2.

def bpll_dbrdq2(q2, wc_obj, par, B, P, lep):
    tauB = par[('lifetime',B)]
    return tauB * bpll_obs(dGdq2_cpaverage, q2, wc_obj, par, B, P, lep)
