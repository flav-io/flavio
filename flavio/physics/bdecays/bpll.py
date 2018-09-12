r"""Functions for exclusive $B\to P\ell^+\ell^-$ decays."""

import flavio
from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc, add_dict
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff, get_wceff_lfv, wctot_dict
from flavio.classes import Observable, Prediction
import warnings

def prefactor(q2, par, B, P):
    GF = par['GF']
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

# form factors
def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity[ff_name].prediction(par_dict=par, wc_obj=None, q2=q2)

# get subleading hadronic contribution
def get_subleading(q2, wc_obj, par_dict, B, P, lep, cp_conjugate):
    if q2 <= 9:
        sub_name = B+'->'+P + 'll subleading effects at low q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
    elif q2 > 14:
        sub_name = B+'->'+P + 'll subleading effects at high q2'
        return AuxiliaryQuantity[sub_name].prediction(par_dict=par_dict, wc_obj=wc_obj, q2=q2, cp_conjugate=cp_conjugate)
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
    N = prefactor(q2, par, B, P)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml, ml, ff, wc_eff, N)
    return h

def helicity_amps(q2, wc_obj, par, B, P, lep):
    if q2 >= 8.7 and q2 < 14 and lep != 'tau':
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, wc_obj, par, B, P, lep, cp_conjugate=False),
        get_subleading(q2, wc_obj, par, B, P, lep, cp_conjugate=False)
        ))

def helicity_amps_bar(q2, wc_obj, par, B, P, lep):
    if q2 >= 8.7 and q2 < 14 and lep != 'tau':
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    return add_dict((
        helicity_amps_ff(q2, wc_obj, par, B, P, lep, cp_conjugate=True),
        get_subleading(q2, wc_obj, par, B, P, lep, cp_conjugate=True)
        ))

def bpll_obs(function, q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    if q2 <= (ml+ml)**2 or q2 > (mB-mP)**2:
        return 0
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    h     = helicity_amps(q2, wc_obj, par, B, P, lep)
    J     = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml, ml)
    if lep == lep:
        h_bar = helicity_amps_bar(q2, wc_obj, par, B, P, lep)
        J_bar = angular.angularcoeffs_general_p(h_bar, q2, mB, mP, mb, 0, ml, ml)
    else:
        # for LFV decays, don't bother about the CP average. There is no strong phase.
        J_bar = J
    return function(J, J_bar)

def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def AFB_num(J):
    return J['b']

def FH_num(J):
    return 2 * (J['a'] + J['c'])

def dGdq2_cpaverage(J, J_bar):
    return (dGdq2(J) + dGdq2(J_bar))/2.
def dGdq2_cpdiff(J, J_bar):
    return (dGdq2(J) - dGdq2(J_bar))/2.

def AFB_cpaverage_num(J, J_bar):
    return (AFB_num(J) + AFB_num(J_bar))/2.

def FH_cpaverage_num(J, J_bar):
    return (FH_num(J) + FH_num(J_bar))/2.

def bpll_obs_int(function, q2min, q2max, wc_obj, par, B, P, lep, epsrel=0.005):
    def obs(q2):
        return bpll_obs(function, q2, wc_obj, par, B, P, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max, epsrel=epsrel)


def bpll_dbrdq2(q2, wc_obj, par, B, P, lep):
    tauB = par['tau_'+B]
    dBR = tauB * bpll_obs(dGdq2_cpaverage, q2, wc_obj, par, B, P, lep)
    if P == 'pi0':
        # factor of 1/2 for neutral pi due to pi = (uubar-ddbar)/sqrt(2)
        return dBR / 2.
    else:
        return dBR

def bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep, epsrel=0.005):
    def obs(q2):
        return bpll_dbrdq2(q2, wc_obj, par, B, P, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max, epsrel=epsrel)/(q2max-q2min)

# Functions returning functions needed for Prediction instances

def bpll_dbrdq2_int_func(B, P, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep)
    return fct

def bpll_dbrdq2_tot_func(B, P, lep):
    def fct(wc_obj, par):
        mB = par['m_'+B]
        mP = par['m_'+P]
        ml = par['m_'+lep]
        ml = par['m_'+lep]
        q2max = (mB-mP)**2
        q2min = (ml+ml)**2
        return bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, lep)*(q2max-q2min)
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

def bpll_obs_int_ratio_leptonflavour(func, B, P, lnum, lden):
    def fct(wc_obj, par, q2min, q2max):
        num = bpll_obs_int(func, q2min, q2max, wc_obj, par, B, P, lnum, epsrel=0.0005)
        if num == 0:
            return 0
        denom = bpll_obs_int(func, q2min, q2max, wc_obj, par, B, P, lden, epsrel=0.0005)
        return num/denom
    return fct

def bpll_obs_ratio_leptonflavour(func, B, P, lnum, lden):
    def fct(wc_obj, par, q2):
        num = bpll_obs(func, q2, wc_obj, par, B, P, lnum)
        if num == 0:
            return 0
        denom = bpll_obs(func, q2, wc_obj, par, B, P, lden)
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
'ACP': {'func_num': dGdq2_cpdiff, 'tex': r'A_\text{CP}', 'desc': 'Direct CP asymmetry'},
'AFB': {'func_num': AFB_cpaverage_num, 'tex': r'A_\text{FB}', 'desc': 'forward-backward asymmetry'},
'FH': {'func_num': FH_cpaverage_num, 'tex': r'F_H', 'desc': 'flat term'},
}
_hadr = {
'B0->K': {'tex': r"B^0\to K^0", 'B': 'B0', 'P': 'K0', },
'B+->K': {'tex': r"B^\pm\to K^\pm ", 'B': 'B+', 'P': 'K+', },
}
_hadr_lfv = {
'B0->K': {'tex': r"\bar B^0\to \bar K^0", 'B': 'B0', 'P': 'K0', },
'B+->K': {'tex': r"B^-\to K^-", 'B': 'B+', 'P': 'K+', },
'B0->pi': {'tex': r"\bar B^0\to \pi^0", 'B': 'B0', 'P': 'pi0', },
'B+->pi': {'tex': r"B^-\to \pi^-", 'B': 'B+', 'P': 'pi+', },
}
_tex_lfv = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-',
    'emu,mue': r'e^\pm\mu^\mp', 'etau,taue': r'e^\pm\tau^\mp',
    'mutau,taumu': r'\mu^\pm\tau^\mp'}

for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():

        _process_tex = _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to P\ell^+\ell^-$ :: $' + _process_tex + r"$"

        for obs in sorted(_observables.keys()):
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _process_tex + r"$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _process_tex + r")$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bpll_obs_int_ratio_func(_observables[obs]['func_num'], dGdq2_cpaverage, _hadr[M]['B'], _hadr[M]['P'], l))

            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _process_tex + r"$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _process_tex + r")$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bpll_obs_ratio_func(_observables[obs]['func_num'], dGdq2_cpaverage, _hadr[M]['B'], _hadr[M]['P'], l))

        # binned branching ratio
        _obs_name = "<dBR/dq2>("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned differential branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\langle \frac{d\text{BR}}{dq^2} \rangle(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bpll_dbrdq2_int_func(_hadr[M]['B'], _hadr[M]['P'], l))

        # differential branching ratio
        _obs_name = "dBR/dq2("+M+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bpll_dbrdq2_func(_hadr[M]['B'], _hadr[M]['P'], l))

        # only for tau: total branching ratio
        if l == 'tau':
            _obs_name = "BR("+M+l+l+")"
            _obs = Observable(name=_obs_name)
            _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
            _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
            _obs.add_taxonomy(_process_taxonomy)
            Prediction(_obs_name, bpll_dbrdq2_tot_func(_hadr[M]['B'], _hadr[M]['P'], l))

# Lepton flavour ratios
for l in [('mu','e'), ('tau','mu'),]:
    for M in _hadr.keys():

        # binned ratio of BRs
        _obs_name = "<R"+l[0]+l[1]+">("+M+"ll)"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Ratio of partial branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
        _obs.tex = r"$\langle R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"} \rangle(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
        for li in l:
            # add taxonomy for both processes (e.g. B->Pee and B->Pmumu)
            _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to P\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
        Prediction(_obs_name, bpll_obs_int_ratio_leptonflavour(dGdq2_cpaverage, _hadr[M]['B'], _hadr[M]['P'], *l))

        # differential ratio of BRs
        _obs_name = "R"+l[0]+l[1]+"("+M+"ll)"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Ratio of differential branching ratios of $" + _hadr[M]['tex'] +_tex[l[0]]+r"^+ "+_tex[l[0]]+r"^-$" + " and " + r"$" + _hadr[M]['tex'] +_tex[l[1]]+r"^+ "+_tex[l[1]]+"^-$")
        _obs.tex = r"$R_{" + _tex[l[0]] + ' ' + _tex[l[1]] + r"}(" + _hadr[M]['tex'] + r"\ell^+\ell^-)$"
        for li in l:
            # add taxonomy for both processes (e.g. B->Pee and B->Pmumu)
            _obs.add_taxonomy(r'Process :: $b$ hadron decays :: FCNC decays :: $B\to P\ell^+\ell^-$ :: $' + _hadr[M]['tex'] +_tex[li]+r"^+"+_tex[li]+r"^-$")
        Prediction(_obs_name, bpll_obs_ratio_leptonflavour(dGdq2_cpaverage, _hadr[M]['B'], _hadr[M]['P'], *l))
