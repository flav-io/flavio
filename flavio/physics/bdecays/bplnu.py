from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff_fccc

r"""Functions for exclusive $B\to P\ell\nu$ decays."""


def prefactor(q2, par, B, P, lep):
    GF = par['Gmu']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    qi_qj = meson_quark[(B, P)]
    if qi_qj == 'bu':
        Vij = ckm.get_ckm(par)[0,2] # V_{ub} for b->u transitions
    if qi_qj == 'bc':
        Vij = ckm.get_ckm(par)[1,2] # V_{cb} for b->c transitions
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*Vij

def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def get_angularcoeff(q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    wc = get_wceff_fccc(q2, wc_obj, par, B, P, lep)
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    qi_qj = meson_quark[(B, P)]
    if qi_qj == 'bu':
        mlight = 0. # neglecting the up quark mass
    if qi_qj == 'bc':
        mlight = running.get_mc(par, scale) # this is needed for scalar contributions
    h = angular.helicity_amps_p(q2, mB, mP, mb, mlight, ml, 0, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, mlight, ml, 0)
    return J

def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def dBRdq2(q2, wc_obj, par, B, P, lep):
    tauB = par['tau_'+B]
    J = get_angularcoeff(q2, wc_obj, par, B, P, lep)
    return tauB * dGdq2(J)
