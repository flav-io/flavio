from math import sqrt,pi
from flavio.physics.bdecays.common import lambda_K, beta_l, meson_quark, meson_ff
from flavio.physics import ckm
from flavio.classes import AuxiliaryQuantity
from flavio.config import config
from flavio.physics.running import running
from flavio.physics.common import conjugate_par, conjugate_wc
from flavio.physics.bdecays import matrixelements, angular
from flavio.physics.bdecays.wilsoncoefficients import get_wceff, wctot_dict
from scipy.integrate import quad
from flavio.classes import Observable, Prediction

"""Functions for exclusive $B\to P\ell^+\ell^-$ decays."""

def prefactor(q2, par, B, P, lep):
    GF = par['Gmu']
    ml = par['m_'+lep]
    scale = config['renormalization scale']['bpll']
    alphaem = running.get_alpha(par, scale)['alpha_e']
    di_dj = meson_quark[(B,P)]
    xi_t = ckm.xi('t',di_dj)(par)
    if q2 <= 4*ml**2:
        return 0
    return 4*GF/sqrt(2)*xi_t*alphaem/(4*pi)

def get_ff(q2, par, B, P):
    ff_name = meson_ff[(B,P)] + ' form factor'
    return AuxiliaryQuantity.get_instance(ff_name).prediction(par_dict=par, wc_obj=None, q2=q2)

def get_angularcoeff(q2, wc, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P, lep)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml, ml, ff, wc, N)
    J = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml, ml)
    return J

def bpll_obs(function, q2, wc_obj, par, B, P, lep):
    ml = par['m_'+lep]
    mB = par['m_'+B]
    mP = par['m_'+P]
    if q2 < 4*ml**2 or q2 > (mB-mP)**2:
        return 0
    scale = config['renormalization scale']['bpll']
    wc = wctot_dict(wc_obj, meson_quark[(B,P)]+lep+lep, scale, par)
    wc_c = conjugate_wc(wc)
    par_c = conjugate_par(par)
    wc_eff = get_wceff(q2, wc, par, B, P, lep, scale)
    wc_eff_c = get_wceff(q2, wc_c, par_c, B, P, lep, scale)
    J     = get_angularcoeff(q2, wc_eff,   par,   B, P, lep)
    J_bar = get_angularcoeff(q2, wc_eff_c, par_c, B, P, lep)
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

def bpll_dbrdq2(q2, wc_obj, par, B, P, lep):
    tauB = par['tau_'+B]
    return tauB * bpll_obs(dGdq2_cpaverage, q2, wc_obj, par, B, P, lep)

# denominator of normalized observables
def denominator(J, J_bar):
    return 2*dGdq2_cpaverage(J, J_bar)

def bpll_obs_int(function, q2min, q2max, wc_obj, par, B, P, lep):
    def obs(q2):
        return bpll_obs(function, q2, wc_obj, par, B, P, lep)
    return quad(obs, q2min, q2max, epsrel=0.01, epsabs=0)[0]

def bpll_obs_int_ratio_func(func_num, func_den, B, P, lep):
    def fct(wc_obj, par, q2min, q2max):
        num = bpll_obs_int(func_num, q2min, q2max, wc_obj, par, B, P, lep)
        if num == 0:
            return 0
        den = bpll_obs_int(func_den, q2min, q2max, wc_obj, par, B, P, lep)
        return num/den
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
'B0->K': {'tex': r"B^0\to K^0", 'B': 'B0', 'V': 'K0', },
'B+->K': {'tex': r"B^+\to K^+", 'B': 'B+', 'V': 'K+', },
}

for l in ['e', 'mu', 'tau']:
    for M in _hadr.keys():
        for obs in sorted(_observables.keys()):
            _obs_name = "<" + obs + ">("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
            _obs.set_description('Binned ' + _observables[obs]['desc'] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$\langle " + _observables[obs]['tex'] + r"\rangle(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bpll_obs_int_ratio_func(_observables[obs]['func_num'], denominator, _hadr[M]['B'], _hadr[M]['V'], l))

            _obs_name = obs + "("+M+l+l+")"
            _obs = Observable(name=_obs_name, arguments=['q2'])
            _obs.set_description(_observables[obs]['desc'][0].capitalize() + _observables[obs]['desc'][1:] + r" in $" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-$")
            _obs.tex = r"$" + _observables[obs]['tex'] + r"(" + _hadr[M]['tex'] +_tex[l]+r"^+"+_tex[l]+"^-)$"
            Prediction(_obs_name, bpll_obs_ratio_func(_observables[obs]['func_num'], denominator, _hadr[M]['B'], _hadr[M]['V'], l))
