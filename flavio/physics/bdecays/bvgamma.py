from math import sqrt,pi
from cmath import exp
import numpy as np
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics import ckm, mesonmixing
from flavio.physics.bdecays.formfactors import FormFactorParametrization as FF
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.common import conjugate_par, conjugate_wc

"""Functions for exclusive $B\to V\gamma$ decays."""

def prefactor(par, B, V):
    mB = par['m_'+B]
    mV = par['m_'+V]
    scale = config['renormalization scale']['bvgamma']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    mb = running.get_mb(par, scale)
    GF = par['Gmu']
    di_dj = meson_quark[(B,V)]
    xi_t = ckm.xi('t',di_dj)(par)
    return ( sqrt((GF**2 * alphaem * mB**3 * mb**2)/(32 * pi**4)
                  * (1-mV**2/mB**2)**3) * xi_t )

def amps(wc, par, B, V):
    N = prefactor(par, B, V)
    qiqj = meson_quark[(B,V)]
    ff = FF.parametrizations['bsz3'].get_ff(meson_ff[(B,V)], 0, par)
    c7 = wc['C7eff_'+qiqj] # e.g. C7eff_bs
    c7p = wc['C7effp_'+qiqj]
    a = {}
    a['L'] = N * c7  * ff['T1']
    a['R'] = N * c7p * ff['T1']
    return a

def amps_bar(wc, par, B, V):
    par_c = conjugate_par(par)
    wc_c = conjugate_wc(wc)
    a = amps(wc_c, par_c, B, V)
    return {'L': a['R'], 'R': a['L']}

def get_a_abar(wc_obj, par, B, V):
    scale = config['renormalization scale']['bvll']
    # these are the b->qee Wilson coefficients - they contain the b->qgamma ones as a subset
    wc = wctot_dict(wc_obj, meson_quark[(B,V)] + 'ee', scale, par)
    a = amps(wc, par, B, V)
    a_bar = amps_bar(wc, par, B, V)
    return a, a_bar

def Gamma(a):
    return ( abs(a['L'])**2 + abs(a['R'])**2 )

def Gamma_CPaverage(a, a_bar):
    return ( Gamma(a) + Gamma(a_bar) )/2.

def BR(wc_obj, par, B, V):
    tauB = par['tau_'+B]
    a, a_bar = get_a_abar(wc_obj, par, B, V)
    return tauB * Gamma_CPaverage(a, a_bar)

def ACP(wc_obj, par, B, V):
    a, a_bar = get_a_abar(wc_obj, par, B, V)
    return ( Gamma(a) - Gamma(a_bar) )/( Gamma(a) + Gamma(a_bar) )


def S(wc_obj, par, B, V):
    a, a_bar = get_a_abar(wc_obj, par, B, V)
    q_over_p = mesonmixing.observables.q_over_p(wc_obj, par, B)
    beta = ckm.get_ckmangle_beta(par)
    num = q_over_p * (a['L'].conj()*a_bar['L']+a['R'].conj()*a_bar['R'])
    den = Gamma_CPaverage(a, a_bar)
    return num.imag / den
