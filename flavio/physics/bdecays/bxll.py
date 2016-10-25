r"""Functions for inclusive $B\to X_q\ell^+\ell^-$ decays."""

import flavio
import numpy as np
from math import pi, log, sqrt
from flavio.math.functions import li2
from flavio.classes import Observable, Prediction
import warnings

def bxll_parameters(par, lep):
    ml = par['m_'+lep]
    scale = flavio.config['renormalization scale']['bxll']
    mb = flavio.physics.running.running.get_mb(par, scale)
    alpha = flavio.physics.running.running.get_alpha(par, scale)
    alpha_s = alpha['alpha_s']
    alpha_e = alpha['alpha_e']
    return ml, mb, alpha_s, alpha_e

def bxll_br_prefactor(par, q, lep):
    ml, mb, alpha_s, alpha_e = bxll_parameters(par, lep)
    bq = 'b' + q
    xi_t = flavio.physics.ckm.xi('t',bq)(par)
    Vcb = flavio.physics.ckm.get_ckm(par)[1,2]
    C = par['C_BXlnu']
    BRSL = par['BR(B->Xcenu)_exp']
    return alpha_e**2/4./pi**2 / mb**2 * BRSL/C * abs(xi_t)**2/abs(Vcb)**2

def inclusive_wc(q2, wc_obj, par, q, lep, mb):
    r"""Returns a dictionary of "inclusive" Wilson coefficients where
    universal bremsstrahlung and virtual corrections have been absorbed."""
    scale = flavio.config['renormalization scale']['bxll']
    alphas = flavio.physics.running.running.get_alpha(par, scale)['alpha_s']
    label = 'b' + q + lep + lep # e.g. bsmumu, bdtautau
    wc = flavio.physics.bdecays.wilsoncoefficients.wctot_dict(wc_obj, label, scale, par, nf_out=5)
    # the "effective" WCs already contain the virtual corrections
    wc_eff = flavio.physics.bdecays.wilsoncoefficients.get_wceff(q2, wc, par, 'B', 'X'+q, lep, scale)
    # add the universal bremsstrahlung corrections to the effective WCs
    brems_7 = 1 + alphas/pi * sigma7(q2/mb**2, scale, mb)
    brems_9 = 1 + alphas/pi * sigma9(q2/mb**2)
    wc_eff['7'] = wc_eff['7'] * brems_7
    wc_eff['v'] = wc_eff['v'] * brems_9
    wc_eff['a'] = wc_eff['a'] * brems_9
    wc_eff['7p'] = wc_eff['7p'] * brems_7
    wc_eff['vp'] = wc_eff['vp'] * brems_9
    wc_eff['ap'] = wc_eff['ap'] * brems_9
    return wc_eff

def _bxll_dbrdq2(q2, wc_obj, par, q, lep):
    ml, mb, alpha_s, alpha_e = bxll_parameters(par, lep)
    if q2 < 4*ml**2 or q2 > mb**2:
        return 0
    N = bxll_br_prefactor(par, q, lep)
    # alphaem = 1/137.035999139 # this is alpha_e(0), a constant for our purposes
    wc_dict = inclusive_wc(q2, wc_obj, par, q, lep, mb)
    sh = q2/mb**2
    mlh = ml/mb
    # for ms=0, rate is simply the sum of C_i and C_i' contributions
    Phi_ll  = ( f_incl(sh, mlh, alpha_s, wc_dict['7'], wc_dict['v'], wc_dict['a'], wc_dict['s'], wc_dict['p'])
         + f_incl(sh, mlh, alpha_s, wc_dict['7p'], wc_dict['vp'], wc_dict['ap'], wc_dict['sp'], wc_dict['pp']) )
    Phi_u = f_sl(alpha_e, alpha_s, par['alpha_s'], mb)
    unc = 1 + par['delta_BX'+q+'ll'] # fudge factor to account for remaining uncertainty
    return N * Phi_ll / Phi_u * unc

def bxll_dbrdq2(q2, wc_obj, par, q, lep):
    r"""Inclusive $B\to  X_q\ell^+\ell^-$ differential branching ratio
    normalized to $B\to X_c\ell\nu$ taken from experiment."""
    if q2 > 8 and q2 < 14:
        warnings.warn("The predictions in the region of narrow charmonium resonances are not meaningful")
    if lep == 'l':
        # average of e and mu!
        return (_bxll_dbrdq2(q2, wc_obj, par, q, 'e') + _bxll_dbrdq2(q2, wc_obj, par, q, 'mu'))/2.
    else:
        return _bxll_dbrdq2(q2, wc_obj, par, q, lep)

def f_incl(sh, mlh, alpha_s, C7t, C9t, C10t, CS, CP):
    # The function of Wilson coefficients and kinematical quantities entering
    # the inclusive $B\to Xll$ rate.
    # NB: CS and CP are defined to be dimensionless!
    return (1-sh)**2 * sqrt(1-4*mlh**2/sh) * (
        # SM-like terms for m_l->0: here we add non-univ. bremsstrahlung corrections
        4*abs(C7t)**2*(1+2/sh) * (1+2*mlh**2/sh) * (1 + alpha_s/pi * tau77(sh))
        + 12*(C7t*C9t.conj()).real * (1+2*mlh**2/sh) * (1 + alpha_s/pi * tau79(sh))
        + (abs(C9t)**2 + abs(C10t)**2) * (1 + 2*sh + 2*mlh**2*(1-sh)/sh) * (1 + alpha_s/pi * tau99(sh))
        # scalar/pseudoscalar operators
        + 3/2. * sh * ((1-4*mlh**2/sh) * abs(CS)**2 + abs(CP)**2)
        # O(m_l) terms
        + 6*mlh**2 * (abs(C9t)**2 - abs(C10t)**2)
        + 6 * mlh * (CP * C10t.conj()).real )


# Functions needed for bremsstrahlung corrections to rate
def sigma9(s):
    return sigma(s) + 3/2
def sigma7(s, scale, mb):
    return sigma(s) + 1/6 - 8/3 * log(scale/mb)
def sigma(s):
    return - 4/3 * li2(s) - 2/3 * log(s) * log(1-s) -2/9 * pi**2 -log(1-s)-2/9 * (1-s) * log(1-s)
def tau77(s):
    return - 2/(9 * (2+s)) * (2 * (1-s)**2 * log(1-s) +(6 * s * (2-2 * s-s**2))/((1-s)**2) * log(s) +(11-7 * s-10 * s**2)/(1-s) )
def tau99(s):
    return -4/(9 * (1+2 * s)) * (2 * (1-s)**2 * log(1-s) +(3 * s * (1+s) * (1-2 * s))/((1-s)**2) * log(s)+(3 * (1-3 * s**2))/(1-s) )
def tau79(s):
    return - (4 * (1-s)**2)/(9 * s) * log(1-s)-(4 * s * (3-2 * s))/(9 * (1-s)**2) * log(s) -(2 * (5-3 * s))/(9 * (1-s))


def f_sl(alpha_e, alpha_s, alpha_s_mu0, mb):
    # semileptonic phase space factor for B->Xulnu
    # see (4.8) of 1503.04849
    ast = alpha_s/4./pi
    k = alpha_e/alpha_s
    eta = alpha_s_mu0/alpha_s
    return ( 1 + ast * (50/3. - 8*pi**2/3.) + k*(12/23.*(1-1/eta)) )

# functions for observables

def bxll_br_int(q2min, q2max, wc_obj, par, q, lep):
    def obs(q2):
        return bxll_dbrdq2(q2, wc_obj, par, q, lep)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max)

def bxll_br_int_func(q, lep):
    def fct(wc_obj, par, q2min, q2max):
        return bxll_br_int(q2min, q2max, wc_obj, par, q, lep)
    return fct

def bxll_dbrdq2_func(q, lep):
    def fct(wc_obj, par, q2):
        return bxll_dbrdq2(q2, wc_obj, par, q, lep)
    return fct

# Observable instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau', 'l': r'\ell'}
for l in ['e', 'mu', 'tau', 'l']:
    for q in ['s', 'd']:

        _process_tex =  r"B\to X_" + q +_tex[l]+r"^+"+_tex[l]+r"^-"
        _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to X\ell^+\ell^-$ :: $' + _process_tex + r"$"

        # binned branching ratio
        _obs_name = "<BR>(B->X"+q+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2min', 'q2max'])
        _obs.set_description(r"Binned branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\langle \text{BR} \rangle(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bxll_br_int_func(q, l))

        # differential branching ratio
        _obs_name = "dBR/dq2(B->X"+q+l+l+")"
        _obs = Observable(name=_obs_name, arguments=['q2'])
        _obs.set_description(r"Differential branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\frac{d\text{BR}}{dq^2}(" + _process_tex + r")$"
        _obs.add_taxonomy(_process_taxonomy)
        Prediction(_obs_name, bxll_dbrdq2_func(q, l))
