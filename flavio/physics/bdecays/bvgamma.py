from math import sqrt,pi
from cmath import exp
import numpy as np
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics import ckm
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running

"""Functions for exclusive $B\to V\gamma$ decays."""

def prefactor(par, B, V):
    mB = par[('mass',B)]
    mV = par[('mass',V)]
    scale = config['bdecays']['scale_bvgamma']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    mb = running.get_mb(par, scale)
    GF = par['Gmu']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    return ( sqrt((GF**2 * alphaem * mB**3 * mb**2)/(32 * pi**4)
                  * (1-mV**2/mB**2)**3) * xi_t )

def amps(wc, par, B, V):
    N = prefactor(par, B, V)
    ff = FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], 0, par)
    c7 = wc['C7eff']
    c7p = wc['C7effp']
    a = {}
    a['L'] = N * c7  * ff['T1']
    a['R'] = N * c7p * ff['T1']
    return a

def amps_bar(wc, par, B, V):
    # FIXME need to implement CP conjugation
    return amps(wc, par, B, V)

def Gamma(a):
    return ( abs(a['L'])**2 + abs(a['R'])**2 )

def Gamma_CPaverage(a, a_bar):
    return ( Gamma(a) + Gamma(a_bar) )/2.

def BR(wc, par, B, V):
    tauB = par[('lifetime',B)]
    a = amps(wc, par, B, V)
    return tauB * Gamma(a)

def BR_CPaverage(wc, par, B, V):
    tauB = par[('lifetime',B)]
    a = amps(wc, par, B, V)
    a_bar = amps_bar(wc, par, B, V)
    return tauB * Gamma_CPaverage(a, a_bar)

def ACP(wc, par, B, V):
    a = amps(wc, par, B, V)
    a_bar = amps_bar(wc, par, B, V)
    return ( Gamma(a) - Gamma(a_bar) )/( Gamma(a) + Gamma(a_bar) )

def S(wc, par, B, V):
    a = amps(wc, par, B, V)
    a_bar = amps_bar(wc, par, B, V)
    beta = ckm.get_ckmangle_beta(par)
    q_over_p  = exp(-2j*beta)
    interf = a['L'].conj()*a_bar['L']+a['R'].conj()*a_bar['R']
    Gav = Gamma_CPaverage(a, a_bar)
    return q_over_p * interf / Gav
