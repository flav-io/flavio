from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, meson_quark, meson_ff
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays import angular

r"""Functions for exclusive $B\to V\ell\nu$ decays."""


def get_wceff(q2, wc_obj, par, B, V, lep):
    scale = config['bdecays']['scale_bvll']
    bqlnu = meson_quark[(B,V)] + lep + 'nu'
    wc = wc_obj.get_wc(bqlnu, scale, par)
    mb = running.get_mb(par, scale)
    c = {}
    c['7']  = 0
    c['7p'] = 0
    c['v']  = (1 + wc['CV_'+bqlnu])/2.
    c['vp'] = (1 + wc['CVp_'+bqlnu])/2.
    c['a']  = -wc['CV_'+bqlnu]/2.
    c['ap'] = -wc['CVp_'+bqlnu]/2.
    c['s']  = 1/2 * mb * wc['CS_'+bqlnu]/2.
    c['sp'] = 1/2 * mb * wc['CSp_'+bqlnu]/2.
    c['p']  = -1/2 * mb * wc['CS_'+bqlnu]/2.
    c['pp'] = -1/2 * mb * wc['CSp_'+bqlnu]/2.
    c['t']  = wc['CT_'+bqlnu]
    c['tp'] = 0
    return c

def get_ff(q2, par, B, V):
    return FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], q2, par)

def prefactor(q2, par, B, V, lep):
    GF = par['Gmu']
    scale = config['bdecays']['scale_bvll']
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    tauB = par[('lifetime',B)]
    laB  = lambda_K(mB**2, mV**2, q2)
    laGa = lambda_K(q2, ml**2, 0.)
    qi_qj = meson_quark[(B, V)]
    if qi_qj == 'bu':
        Vij = ckm.get_ckm(par)[0,2] # V_{ub} for b->u transitions
    if qi_qj == 'bc':
        Vij = ckm.get_ckm(par)[1,2] # V_{cb} for b->c transitions
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*Vij

def get_angularcoeff(q2, wc_obj, par, B, V, lep):
    scale = config['bdecays']['scale_bvlnu']
    wc = get_wceff(q2, wc_obj, par, B, V, lep)
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    mb = running.get_mb(par, scale)
    qi_qj = meson_quark[(B, V)]
    if qi_qj == 'bu':
        mlight = 0. # neglecting the up quark mass
    if qi_qj == 'bc':
        mlight = running.get_mc(par, scale) # this is needed for scalar contributions
    N = prefactor(q2, par, B, V, lep)
    ff = get_ff(q2, par, B, V)
    h = angular.helicity_amps_v(q2, mB, mV, mb, mlight, ml, 0, ff, wc, N)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, mlight, ml, 0)
    return J

def dGdq2(J):
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def dBRdq2(q2, wc_obj, par, B, V, lep):
    tauB = par[('lifetime',B)]
    J = get_angularcoeff(q2, wc_obj, par, B, V, lep)
    return tauB * dGdq2(J)
