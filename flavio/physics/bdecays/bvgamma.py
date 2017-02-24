r"""Functions for exclusive $B\to V\gamma$ decays."""

from math import sqrt,pi
from cmath import exp
import numpy as np
import flavio
from flavio.physics.bdecays.common import meson_quark, meson_ff
from flavio.physics import ckm, mesonmixing
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict, get_wceff
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

def prefactor_helicityamps(q2, par, B, V):
    N = prefactor(par, B, V)
    N_BVll = flavio.physics.bdecays.bvll.amplitudes.prefactor(q2, par, B, V)
    mB = par['m_'+B]
    mV = par['m_'+V]
    laB = flavio.physics.bdecays.common.lambda_K(mB**2, mV**2, q2)
    scale = config['renormalization scale']['bvgamma']
    mb = flavio.physics.running.running.get_mb(par, scale)
    return N/(+1j * mb/q2 * sqrt(laB) * 2)/N_BVll


def amps_ff(wc_obj, par_dict, B, V, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    N = prefactor(par, B, V)
    bq = meson_quark[(B,V)]
    ff_name = meson_ff[(B,V)] + ' form factor'
    ff = AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=0.)
    scale = config['renormalization scale']['bvgamma']
    # these are the b->qee Wilson coefficients - they contain the b->qgamma ones as a subset
    wc = wctot_dict(wc_obj, bq+'ee', scale, par)
    if cp_conjugate:
        wc = conjugate_wc(wc)
    delta_C7 = flavio.physics.bdecays.matrixelements.delta_C7(par=par, wc=wc, q2=0, scale=scale, qiqj=bq)
    a = {}
    a['L'] = N * (wc['C7eff_'+bq] + delta_C7)  * ff['T1']
    a['R'] = N * wc['C7effp_'+bq] * ff['T1']
    return a

def amps_ss(wc_obj, par, B, V, cp_conjugate):
    scale = config['renormalization scale']['bvgamma']
    ss_name = B+'->'+V+'ll spectator scattering'
    q2=0.001 # away from zero to avoid pole
    amps = AuxiliaryQuantity[ss_name].prediction(par_dict=par, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    N = prefactor_helicityamps(q2, par, B, V)
    a = {}
    a['L'] = -N * amps[('mi' ,'V')]
    a['R'] = +N * amps[('pl' ,'V')]
    return a

def amps_subleading(wc_obj, par, B, V, cp_conjugate):
    scale = config['renormalization scale']['bvgamma']
    sub_name = B+'->'+V+ 'll subleading effects at low q2'
    q2=0.001 # away from zero to avoid pole
    amps = AuxiliaryQuantity[sub_name].prediction(par_dict=par, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    N = prefactor_helicityamps(q2, par, B, V)
    a = {}
    a['L'] = -N * amps[('mi' ,'V')]
    a['R'] = +N * amps[('pl' ,'V')]
    return a

def amps(*args, **kwargs):
    return add_dict((
        amps_ff(*args, cp_conjugate=False, **kwargs),
        amps_ss(*args, cp_conjugate=False, **kwargs),
        amps_subleading(*args, cp_conjugate=False, **kwargs),
        ))

def amps_bar(*args, **kwargs):
    a = add_dict((
        amps_ff(*args, cp_conjugate=True, **kwargs),
        amps_ss(*args, cp_conjugate=True, **kwargs),
        amps_subleading(*args, cp_conjugate=True, **kwargs),
        ))
    return {'L': a['R'], 'R': a['L']}

def get_a_abar(wc_obj, par, B, V):
    scale = config['renormalization scale']['bvll']
    a = amps(wc_obj, par, B, V)
    a_bar = amps_bar(wc_obj, par, B, V)
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

def S_A_complex(wc_obj, par, B, V):
    a, a_bar = get_a_abar(wc_obj, par, B, V)
    q_over_p = mesonmixing.observables.q_over_p(wc_obj, par, B)
    beta = ckm.get_ckmangle_beta(par)
    den = Gamma_CPaverage(a, a_bar)
    # minus sign from different convention of q/p compared to Ball/Zwicky
    return -q_over_p * (a['L']*a_bar['L'].conj()+a['R']*a_bar['R'].conj())/den

def S(wc_obj, par, B, V):
    return S_A_complex(wc_obj, par, B, V).imag

def A_DeltaGamma(wc_obj, par, B, V):
    return S_A_complex(wc_obj, par, B, V).real

def BR_timeint(wc_obj, par, B, V):
    A = A_DeltaGamma(wc_obj, par, B, V)
    BR0 = BR(wc_obj, par, B, V)
    y = par['DeltaGamma/Gamma_'+B]/2.
    return (1 - A*y)/(1-y**2) * BR0

def BVgamma_function(function, B, V):
    return lambda wc_obj, par: function(wc_obj, par, B, V)


# Observable and Prediction instances

_func = {'BR': BR, 'ACP': ACP}
_tex = {'BR': r'\text{BR}', 'ACP': r'A_{CP}'}
_desc = {'BR': 'Branching ratio', 'ACP': 'Direct CP asymmetry'}

_process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to V\gamma$ :: $'


for key in _func.keys():
    _obs_name = key + "(B+->K*gamma)"
    _obs = Observable(_obs_name)
    _process_tex = r"B^+\to K^{*+}\gamma"
    _obs.set_description(_desc[key] + r" of $" + _process_tex + r"$")
    _obs.tex = r'$' + _tex[key] + r"(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
    Prediction(_obs_name, BVgamma_function(_func[key], 'B+', 'K*+'))

    _obs_name = key + "(B0->K*gamma)"
    _obs = Observable(_obs_name)
    _process_tex = r"B^0\to K^{*0}\gamma"
    _obs.set_description(_desc[key] + r" of $" + _process_tex + r"$")
    _obs.tex = r'$' + _tex[key] + r"(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
    Prediction(_obs_name, BVgamma_function(_func[key], 'B0', 'K*0'))

_obs_name = "ACP(Bs->phigamma)"
_obs = Observable(_obs_name)
_process_tex = r"B_s\to \phi\gamma"
_obs.set_description(_desc['ACP'] + r" of $" + _process_tex + r"$")
_obs.tex = r'$' + _tex['ACP'] + r"(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BVgamma_function(_func['ACP'], 'Bs', 'phi'))

_obs_name = "BR(Bs->phigamma)"
_obs = Observable(_obs_name)
_process_tex = r"B_s\to \phi\gamma"
_obs.set_description(r"Time-integrated branching ratio of $" + _process_tex + r"$")
_obs.tex = r"$\overline{\text{BR}}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BVgamma_function(BR_timeint, 'Bs', 'phi'))

_obs_name = "ADeltaGamma(Bs->phigamma)"
_obs = Observable(_obs_name)
_process_tex = r"B_s\to \phi\gamma"
_obs.set_description(r"Mass-eigenstate rate asymmetry in $" + _process_tex + r"$")
_obs.tex = r"$A_{\Delta\Gamma}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BVgamma_function(A_DeltaGamma, 'Bs', 'phi'))

_obs_name = "S_K*gamma"
_obs = Observable(_obs_name)
_process_tex = r"B^0\to K^{*0}\gamma"
_obs.set_description(r"Mixing-induced CP asymmetry in $" + _process_tex + r"$")
_obs.tex = r'$S_{K^{*}\gamma}$'
_obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BVgamma_function(S, 'B0', 'K*0'))

_obs_name = "S_phigamma"
_obs = Observable(_obs_name)
_process_tex = r"B_s\to \phi\gamma"
_obs.set_description(r"Mixing-induced CP asymmetry in $" + _process_tex + r"$")
_obs.tex = r'$S_{\phi\gamma}$'
_obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
Prediction(_obs_name, BVgamma_function(S, 'Bs', 'phi'))
