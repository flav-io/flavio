r"""Functions for exclusive $B\to P\ell^+\ell^-$ decays."""

from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff, wctot_dict
from scipy.integrate import quad
from flavio.classes import Observable, Prediction
import warnings

def prefactor(q2, par, B, P, lep):
    GF = par['GF']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

# form factors
def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

# get subleading hadronic contribution
def get_subleading(q2, wc_obj, par_dict, B, P, lep, cp_conjugate):
    if q2 <= 9:
        sub_name = B+'->'+P+lep+lep + ' subleading effects at low q2'
        return AuxiliaryQuantity.get_instance(sub_name).prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    elif q2 > 14:
        sub_name = B+'->'+P+lep+lep + ' subleading effects at high q2'
        return AuxiliaryQuantity.get_instance(sub_name).prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    else:
        return {}


def helicity_amps_ff(q2, wc_obj, par_dict, B, P, lep, cp_conjugate):
    par = par_dict.copy()
    if cp_conjugate:
        par = conjugate_par(par)
    scale = config['renormalization scale']['bpll']
    label = meson_quark[(B,P)] + lep + lep # e.g. bsmumu, bdtautau
    wc = wctot_dict(wc_obj, label, scale, par)
    if cp_conjugate:
        wc = conjugate_wc(wc)
    wc_eff = get_wceff(q2, wc, par, B, P, lep, scale)
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml, ml, ff, wc_eff, N)
    return h

def helicity_amps(q2, wc_obj, par, B, P, lep):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, wc_obj, par, B, P, lep, cp_conjugate=False),
        get_subleading(q2, wc_obj, par, B, P, lep, cp_conjugate=False)
        ))

def helicity_amps_bar(q2, wc_obj, par, B, P, lep):
    if q2 >= 8.7 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, wc_obj, par, B, P, lep, cp_conjugate=True),
        get_subleading(q2, wc_obj, par, B, P, lep, cp_conjugate=True)
        ))

def bpll_obs(function, q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    if q2 < 4*ml**2 or q2 > (mB-mP)**2:
        return 0
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    h     = helicity_amps(q2, wc_obj, par, B, P, lep)
    h_bar = helicity_amps_bar(q2, wc_obj, par, B, P, lep)
    J     = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml, ml)
    J_bar = angular.angularcoeffs_general_p(h_bar, q2, mB, mP, mb, 0, ml, ml)
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


# denominator of normalized observables
def denominator(J, J_bar):
    return 2*dGdq2_cpaverage(J, J_bar)

def bpll_obs_int(function, q2min, q2max, wc_obj, par, B, P, lep):
    def obs(q2):
        return bpll_obs(function, q2, wc_obj, par, B, P, lep)
    return quad(obs, q2min, q2max, epsrel=0.01, epsabs=0)[0]


def bpll_dbrdq2(q2, wc_obj, par, B, P, lep):
    tauB = par['tau_'+B]
    return tauB * bpll_obs(dGdq2_cpaverage, q2, wc_obj, par, B, P, lep)

def bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep):
    def obs(q2):
        return bpll_dbrdq2(q2, wc_obj, par, B, P, lep)
    return quad(obs, q2min, q2max, epsrel=0.01, epsabs=0)[0]/(q2max-q2min)

# Functions returning functions needed for Prediction instances

def bpll_dbrdq2_int_func(B, P, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep)
    return fct

def bpll_dbrdq2_func(B, P, lep):
    def fct(wc_obj, par, q2):
        return bpll_dbrdq2(q2, wc_obj, par, B, P, lep)
    return fct

def bpll_obs_int_ratio_func(func_num, func_den, B, P, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bpll_obs_int(func_num, q2min, q2max, wc_obj, par, B, P, lep)
        if num == 0:
            return 0
        den = bpll_obs_int(func_den, q2min, q2max, wc_obj, par, B, P, lep)
        return num/den
    return fct

def bpll_obs_int_ratio_leptonflavour(func, B, P, l1, l2):
    def fct(wc_obj, par, q2min, q2max):
        num = bpll_obs_int(func, q2min, q2max, wc_obj, par, B, P, l1)
        if num == 0:
            return 0
        denom = bpll_obs_int(func, q2min, q2max, wc_obj, par, B, P, l2)
        return num/denom
    return fct


def bpll_obs_ratio_func(func_num, func_den, B, P, lep):
    def fct(wc_obj, par, q2):
        num = bpll_obs(func_num, q2, wc_obj, par, B, P, lep)
        if num == 0:
            return 0
        den = bpll_obs(func_den, q2, wc_obj, par, B, P, lep)
        return num/den
    return fct

# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
_observables = {
'AFB': {'func_num': AFB_cpaverage_num, 'tex': r'A_\text{FB}', 'desc': 'forward-backward asymmetry'},
'FH': {'func_num': FH_cpaverage_num, 'tex': r'F_H', 'desc': 'flat term'},
}
_hadr = {
'B0->K': {'tex': r"B^0\to K^0", 'B': 'B0', 'P': 'K0', },
'B+->K': {'tex': r"B^+\to K^+", 'B': 'B+', 'P': 'K+', },
}

for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():
        for obs in sorted(_observables.keys()):
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bpll_obs_int_ratio_func(_observables[obs]['func_num'], denominator, _hadr[M]['B'], _hadr[M]['P'], l))

            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bpll_obs_ratio_func(_observables[obs]['func_num'], denominator, _hadr[M]['B'], _hadr[M]['P'], l))

        # binned branching ratio
        _obs_name = "<dBR/dq2>("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        Prediction(_obs_name, bpll_dbrdq2_int_func(_hadr[M]['B'], _hadr[M]['P'], l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential branching ratio of $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
        _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
        Prediction(_obs_name, bpll_dbrdq2_func(_hadr[M]['B'], _hadr[M]['P'], l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():

            # binned ratio of BRs
            _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
            _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
            Prediction(_obs_name, bpll_obs_int_ratio_leptonflavour(dGdq2_cpaverage, _hadr[M]['B'], _hadr[M]['P'], *l))
