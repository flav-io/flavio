from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running
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
    ml = par[('mass',lep)]
    mB = par[('mass',B)]
    scale = config['bdecays']['scale_bpll']
    mb = running.get_mb(par, scale)
    mP = par[('mass',P)]
    c7pl = wc['C7eff'] + wc['C7effp']
    c9pl = wc['C9'] + wc['C9p']
    c10pl = wc['C10'] + wc['C10p']
    cspl = wc['CS'] + wc['CSp']
    cppl = wc['CP'] + wc['CPp']
    N = prefactor(q2, par, B, P, lep)
    ff = FF.parametrizations['btop_lattice'].get_ff(meson_ff[(B,P)], q2, par)
    F = {}
    F['A'] = N * c10pl * ff['f+']
    F['V'] = N * ( c9pl * ff['f+'] + c7pl * ff['fT'] * 2*mb/(mB+mP) )
    F['P'] = (- N * ml * c10pl * (ff['f+'] + (ff['f+']-ff['f0'])*((mB**2 - mP**2)/q2))
               + N * cppl * ff['f0'] * (mB**2 - mP**2)/mb/2.)
    F['S'] = N * cspl * ff['f0'] * (mB**2 - mP**2)/mb/2.
    return F

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

def dGdq2(ac):
    return 2 * (ac['a'] + ac['c']/3.)

def AFB(ac):
    return ac['b']

def FH(ac):
    return 2 * (ac['a'] + ac['c'])

def bpll_obs(function, q2, wc, par, B, P, lep):
    a = amps(q2, wc, par, B, P, lep)
    ac = angulardist(a, q2, par, B, P, lep)
    return function(ac)

def bpll_dbrdq2(q2, wc, par, B, P, lep):
    tauB = par[('lifetime',B)]
    return tauB * bpll_obs(dGdq2, q2, wc, par, B, P, lep)
