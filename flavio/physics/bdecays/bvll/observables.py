"""Functions for exclusive $B\to V\ell^+\ell^-$ decays."""

from math import sqrt,pi
import numpy as np
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict
from flavio.physics.bdecays import matrixelements
from flavio.physics import ckm
from flavio.config import config
from flavio.physics.running import running
from .amplitudes import *
from scipy.integrate import quad
from flavio.classes import Observable, Prediction
import flavio

def dGdq2(J):
    return 3/4. * (2 * J['1s'] + J['1c']) - 1/4. * (2 * J['2s'] + J['2c'])

def dGdq2_ave(J, J_bar):
    return ( dGdq2(J) + dGdq2(J_bar) )/2.

def dGdq2_diff(J, J_bar):
    return ( dGdq2(J) - dGdq2(J_bar) )/2.

# denominator of S_i and A_i observables
def SA_den(J, J_bar):
    return 2*dGdq2_ave(J, J_bar)

# denominator of P_i observables
def P_den(J, J_bar):
    return S_theory_num(J, J_bar, '2s')

def S_theory(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the theory convention."""
    return S_theory_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def S_theory_num(J, J_bar, i):
    return (J[i] + J_bar[i])

def A_theory(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the theory convention."""
    return A_theory_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def A_theory_num(J, J_bar, i):
    return (J[i] - J_bar[i])

def S_experiment(J, J_bar, i):
    r"""CP-averaged angular observable $S_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return S_experiment_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def S_experiment_num(J, J_bar, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -S_theory_num(J, J_bar, i)
    return S_theory_num(J, J_bar, i)

def A_experiment(J, J_bar, i):
    r"""Angular CP asymmetry $A_i$ in the LHCb convention.

    See eq. (C.8) of arXiv:1506.03970v2.
    """
    return A_experiment_num(J, J_bar, i)/SA_den(J, J_bar)

# numerator
def A_experiment_num(J, J_bar, i):
    if i in [4, '6s', '6c', 7, 9]:
        return -A_theory_num(J, J_bar, i)
    return A_theory_num(J, J_bar, i)

def AFB_experiment(J, J_bar):
    r"""Forward-backward asymmetry in the LHCb convention.

    See eq. (C.9) of arXiv:1506.03970v2.
    """
    return AFB_experiment_num(J, J_bar)/SA_den(J, J_bar)

def AFB_experiment_num(J, J_bar):
    return 3/4.*S_experiment_num(J, J_bar, '6s')

def AFB_theory(J, J_bar):
    """Forward-backward asymmetry in the original theory convention.
    """
    return AFB_theory_num(J, J_bar)/SA_den(J, J_bar)

def AFB_theory_num(J, J_bar):
    return 3/4.*S_theory_num(J, J_bar, '6s')

def FL(J, J_bar):
    r"""Longitudinal polarization fraction $F_L$"""
    return FL_num(J, J_bar)/SA_den(J, J_bar)

def FL_num(J, J_bar):
    return -S_theory_num(J, J_bar, '2c')

def FLhat(J, J_bar):
    r"""Modified longitudinal polarization fraction for vanishing lepton masses,
    $\hat F_L$.

    See eq. (32) of arXiv:1510.04239.
    """
    return FLhat_num(J, J_bar)/SA_den(J, J_bar)

def FLhat_num(J, J_bar):
    return -S_theory_num(J, J_bar, '1c')


def bvll_obs(function, q2, wc_obj, par, B, V, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mV = par['m_'+V]
    if q2 < 4*ml**2 or q2 > (mB-mV)**2:
        return 0
    scale = config['renormalization scale']['bvll']
    mb = running.get_mb(par, scale)
    ff = get_ff(q2, par, B, V)
    h = helicity_amps(q2, wc_obj, par, B, V, lep)
    h_bar = helicity_amps_bar(q2, wc_obj, par, B, V, lep)
    J = angular.angularcoeffs_general_v(h, q2, mB, mV, mb, 0, ml, ml)
    J_bar = angular.angularcoeffs_general_v(h_bar, q2, mB, mV, mb, 0, ml, ml)
    return function(J, J_bar)

def bvll_dbrdq2(q2, wc_obj, par, B, V, lep):
    tauB = par['tau_'+B]
    return tauB * bvll_obs(dGdq2_ave, q2, wc_obj, par, B, V, lep)

def bvll_obs_int(function, q2min, q2max, wc_obj, par, B, V, lep):
    def obs(q2):
        return bvll_obs(function, q2, wc_obj, par, B, V, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)

def bvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep):
    def obs(q2):
        return bvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)/(q2max-q2min)

# Functions returning functions needed for Prediction instances

def bvll_dbrdq2_int_func(B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bvll_dbrdq2_int(q2min, q2max, wc_obj, par, B, V, lep)
    return fct

def bvll_dbrdq2_func(B, V, lep):
    def fct(wc_obj, par, q2):
        return bvll_dbrdq2(q2, wc_obj, par, B, V, lep)
    return fct

def bvll_obs_int_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bvll_obs_int(func_num, q2min, q2max, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bvll_obs_int(func_den, q2min, q2max, wc_obj, par, B, V, lep)
        return num/denom
    return fct

def bvll_obs_int_ratio_leptonflavour(func, B, V, l1, l2):
    def fct(wc_obj, par, q2min, q2max):
        num = bvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l1)
        if num == 0:
            return 0
        denom = bvll_obs_int(func, q2min, q2max, wc_obj, par, B, V, l2)
        return num/denom
    return fct

def bvll_obs_ratio_func(func_num, func_den, B, V, lep):
    def fct(wc_obj, par, q2):
        num = bvll_obs(func_num, q2, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom = bvll_obs(func_den, q2, wc_obj, par, B, V, lep)
        return num/denom
    return fct

# function needed for the P' "optimized" observables
# note that this is the convention used by LHCb, NOT the one used in 1303.5794!
def bvll_pprime_func(func_num, B, V, lep):
    def fct(wc_obj, par, q2):
        num = bvll_obs(func_num, q2, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom_2s = bvll_obs(lambda J, J_bar: S_experiment_num(J, J_bar, '2s'), q2, wc_obj, par, B, V, lep)
        denom_2c = bvll_obs(lambda J, J_bar: S_experiment_num(J, J_bar, '2c'), q2, wc_obj, par, B, V, lep)
        denom = 2*sqrt(-denom_2s*denom_2c)
        return num/denom
    return fct

def bvll_pprime_int_func(func_num, B, V, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bvll_obs_int(func_num, q2min, q2max, wc_obj, par, B, V, lep)
        if num == 0:
            return 0
        denom_2s = bvll_obs_int(lambda J, J_bar: S_experiment_num(J, J_bar, '2s'), q2min, q2max, wc_obj, par, B, V, lep)
        denom_2c = bvll_obs_int(lambda J, J_bar: S_experiment_num(J, J_bar, '2c'), q2min, q2max, wc_obj, par, B, V, lep)
        denom = 2*sqrt(-denom_2s*denom_2c)
        return num/denom
    return fct


# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'ACP': {'func_num': dGdq2_diff, 'tex': r'A_\text{CP}', 'desc': 'Direct CP asymmetry'},
'AFB': {'func_num': AFB_experiment_num, 'tex': r'A_\text{FB}', 'desc': 'forward-backward asymmetry'},
'FL': {'func_num': FL_num, 'tex': r'F_L', 'desc': 'longitudinal polarization fraction'},
'S3': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 3), 'tex': r'S_3', 'desc': 'CP-averaged angular observable'},
'S4': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 4), 'tex': r'S_4', 'desc': 'CP-averaged angular observable'},
'S5': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 5), 'tex': r'S_5', 'desc': 'CP-averaged angular observable'},
'S7': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 7), 'tex': r'S_7', 'desc': 'CP-averaged angular observable'},
'S8': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 8), 'tex': r'S_8', 'desc': 'CP-averaged angular observable'},
'S9': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 9), 'tex': r'S_9', 'desc': 'CP-averaged angular observable'},
'A3': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 3), 'tex': r'A_3', 'desc': 'Angular CP asymmetry'},
'A4': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 4), 'tex': r'A_4', 'desc': 'Angular CP asymmetry'},
'A5': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 5), 'tex': r'A_5', 'desc': 'Angular CP asymmetry'},
'A6s': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, '6s'), 'tex': r'A_6^s', 'desc': 'Angular CP asymmetry'},
'A7': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 7), 'tex': r'A_7', 'desc': 'Angular CP asymmetry'},
'A8': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 8), 'tex': r'A_8', 'desc': 'Angular CP asymmetry'},
'A9': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 9), 'tex': r'A_9', 'desc': 'Angular CP asymmetry'},
}
# for the P observables, the convention of LHCb is used. This differs by a
# sign in P_2 and P_3 from the convention in arXiv:1303.5794
_observables_p = {
'P1': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 3)/2., 'tex': r'P_1', 'desc': "CP-averaged \"optimized\" angular observable"},
'P2': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, '6s')/8., 'tex': r'P_2', 'desc': "CP-averaged \"optimized\" angular observable"},
'P3': {'func_num': lambda J, J_bar: -S_experiment_num(J, J_bar, 9)/4., 'tex': r'P_3', 'desc': "CP-averaged \"optimized\" angular observable"},
'ATIm': {'func_num': lambda J, J_bar: A_experiment_num(J, J_bar, 9)/2., 'tex': r'A_T^\text{Im}', 'desc': "Transverse CP asymmetry"},
}
_observables_pprime = {
'P4p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 4), 'tex': r'P_4^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
'P5p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 5), 'tex': r'P_5^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
# yes, P6p depends on J_7, not J_6. Don't ask why.
'P6p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 7), 'tex': r'P_6^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
'P8p': {'func_num': lambda J, J_bar: S_experiment_num(J, J_bar, 8), 'tex': r'P_8^\prime', 'desc': "CP-averaged \"optimized\" angular observable"},
}
_hadr = {
'B0->K*': {'tex': r"B^0\to K^{\ast 0}", 'B': 'B0', 'V': 'K*0', },
'B+->K*': {'tex': r"B^+\to K^{\ast +}", 'B': 'B+', 'V': 'K*+', },
}

for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():
        for obs in sorted(_observables.keys()):

            # binned angular observables
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_obs_int_ratio_func(_observables[obs]['func_num'], SA_den, _hadr[M]['B'], _hadr[M]['V'], l))

            # differential angular observables
            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_obs_ratio_func(_observables[obs]['func_num'], SA_den, _hadr[M]['B'], _hadr[M]['V'], l))

        for obs in sorted(_observables_p.keys()):

            # binned "optimized" angular observables P
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables_p[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables_p[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_obs_int_ratio_func(_observables_p[obs]['func_num'], P_den, _hadr[M]['B'], _hadr[M]['V'], l))

            # differential "optimized"  angular observables
            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables_p[obs]['desc'][0].capitalize() + _observables_p[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables_p[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_obs_ratio_func(_observables_p[obs]['func_num'], P_den, _hadr[M]['B'], _hadr[M]['V'], l))

        for obs in sorted(_observables_pprime.keys()):

            # binned "optimized" angular observables P'
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables_pprime[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables_pprime[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_pprime_int_func(_observables_pprime[obs]['func_num'], _hadr[M]['B'], _hadr[M]['V'], l))

            # differential "optimized"  angular observables
            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables_pprime[obs]['desc'][0].capitalize() + _observables_pprime[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables_pprime[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bvll_pprime_func(_observables_pprime[obs]['func_num'], _hadr[M]['B'], _hadr[M]['V'], l))

        # binned branching ratio
        _obs_name = "<dBR/dq2>("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        Prediction(_obs_name, bvll_dbrdq2_int_func(_hadr[M]['B'], _hadr[M]['V'], l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        Prediction(_obs_name, bvll_dbrdq2_func(_hadr[M]['B'], _hadr[M]['V'], l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():

            # binned ratio of BRs
            _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
            _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
            Prediction(_obs_name, bvll_obs_int_ratio_leptonflavour(dGdq2_ave, _hadr[M]['B'], _hadr[M]['V'], *l))
