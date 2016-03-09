r"""Functions for exclusive $B\to V\gamma$ decays."""

from math import sqrt,pi
from cmath import exp
import numpy as np
import flavio
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics import ckm, mesonmixing
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.classes import AuxiliaryQuantity, Observable, Prediction


def prefactor(par, B, V):
    mB = par['m_'+B]
    mV = par['m_'+V]
    scale = config['renormalization scale']['bvgamma']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    mb = running.get_mb(par, scale)
    GF = par['GF']
    bq = meson_quark[(B,V)]
    xi_t = ckm.xi('t',bq)(par)
    return ( sqrt((GF**2 * alphaem * mB**3 * mb**2)/(32 * pi**4)
                  * (1-mV**2/mB**2)**3) * xi_t )

def amps_ff(wc, par, B, V):
    N = prefactor(par, B, V)
    bq = meson_quark[(B,V)]
    ff_name = meson_ff[(B,V)] + ' form factor'
    ff = AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=0.)
    scale = config['renormalization scale']['bvgamma']
    bq = meson_quark[(B,V)]
    delta_C7 = flavio.physics.bdecays.matrixelements.delta_C7(par=par, wc=wc, q2=0, scale=scale, qiqj=bq)
    c7 = wc['C7eff_'+bq] + delta_C7
    c7p = wc['C7effp_'+bq]
    a = {}
    a['L'] = N * c7  * ff['T1']
    a['R'] = N * c7p * ff['T1']
    return a

def amps_qcdf(wc, par, B, V):
    N = prefactor(par, B, V)
    scale = config['renormalization scale']['bvgamma']
    T_perp = flavio.physics.bdecays.bvll.qcdf.T_perp(q2=0, par=par, wc=wc, B=B, V=V, scale=scale)
    a = {}
    a['L'] = N * T_perp
    a['R'] = 0
    return a

def amps(*args, **kwargs):
    return add_dict((
        amps_ff(*args, **kwargs),
        amps_qcdf(*args, **kwargs),
        ))

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
    # minus sign from different convention of q/p compared to Ball/Zwicky
    num = -q_over_p * (a['L'].conj()*a_bar['L']+a['R'].conj()*a_bar['R'])
    den = Gamma_CPaverage(a, a_bar)
    return num.imag / den


def BVgamma_function(function, B, V):
    return lambda wc_obj, par: function(wc_obj, par, B, V)


# Observable and Prediction instances

_func = {'BR': BR, 'ACP': ACP}
_tex = {'BR': r'\text{BR}', 'ACP': r'A_{CP}'}
_desc = {'BR': 'Branching ratio', 'ACP': 'Direct CP asymmetry'}

for key in _func.keys():
    _obs_name = key + "(B+->K*gamma)"
    _obs = Observable(_obs_name)
    _obs.set_description(_desc[key] + r" of $B^+\to K^{*+}\gamma$")
    _obs.tex = r'$' + _tex[key] + r"(B^+\to K^{*+}\gamma)$"
    Prediction(_obs_name, BVgamma_function(_func[key], 'B+', 'K*+'))

    _obs_name = key + "(B0->K*gamma)"
    _obs = Observable(_obs_name)
    _obs.set_description(_desc[key] + r" of $B^0\to K^{*0}\gamma$")
    _obs.tex = r'$' + _tex[key] + r"(B^0\to K^{*0}\gamma)$"
    Prediction(_obs_name, BVgamma_function(_func[key], 'B0', 'K*0'))

_obs_name = "S_K*gamma"
_obs = Observable(_obs_name)
_obs.set_description(r"Mixing-induced CP asymmetry in $B^0\to K^{*0}\gamma$")
_obs.tex = r'$S_{K^{*}\gamma}$'
Prediction(_obs_name, BVgamma_function(S, 'B0', 'K*0'))
