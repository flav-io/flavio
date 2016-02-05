from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff, wctot_dict

"""Functions for exclusive $B\to P\ell^+\ell^-$ decays."""

def prefactor(q2, par, B, P, lep):
    GF = par['Gmu']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def get_angularcoeff(q2, wc, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml, ml, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml, ml)
    return J

def bpll_obs(function, q2, wc_obj, par, B, P, lep):
    scale = config['renormalization scale']['bpll']
    wc = wctot_dict(wc_obj, meson_quark[(B,P)]+lep+lep, scale, par)
    wc_c = conjugate_wc(wc)
    par_c = conjugate_par(par)
    wc_eff = get_wceff(q2, wc, par, B, P, lep, scale)
    wc_eff_c = get_wceff(q2, wc_c, par_c, B, P, lep, scale)
    J     = get_angularcoeff(q2, wc_eff,   par,   B, P, lep)
    J_bar = get_angularcoeff(q2, wc_eff_c, par_c, B, P, lep)
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
    tauB = par['tau_'+B]
    return tauB * bpll_obs(dGdq2_cpaverage, q2, wc_obj, par, B, P, lep)
