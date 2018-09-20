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
from flavio.physics.bdecays.bpll import prefactor, get_ff
import warnings


def helicity_amps(q2, wc, par_dict, B, P, l1, l2):
    par = par_dict.copy()
    scale = config['renormalization scale']['bpll']
    wc_eff = get_wceff_lfv(q2, wc, par, B, P, l1, l2, scale)
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    mP = par['m_'+P]
    mb = running.get_mb(par, scale)
    N = prefactor(q2, par, B, P)
    ff = get_ff(q2, par, B, P)
    h = angular.helicity_amps_p(q2, mB, mP, mb, 0, ml1, ml2, ff, wc_eff, N)
    return h


def bpll_obs(function, q2, wc, par, B, P, l1, l2):
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_'+B]
    mP = par['m_'+P]
    if q2 <= (ml1+ml2)**2 or q2 > (mB-mP)**2:
        return 0
    scale = config['renormalization scale']['bpll']
    mb = running.get_mb(par, scale)
    h     = helicity_amps(q2, wc, par, B, P, l1, l2)
    J     = angular.angularcoeffs_general_p(h, q2, mB, mP, mb, 0, ml1, ml2)
    return function(J)

def dGdq2(J):
    return 2 * (J['a'] + J['c']/3.)

def bpll_dbrdq2(q2, wc, par, B, P, l1, l2):
    tauB = par['tau_'+B]
    dBR = tauB * bpll_obs(dGdq2, q2, wc, par, B, P, l1, l2)
    if P == 'pi0':
        # factor of 1/2 for neutral pi due to pi = (uubar-ddbar)/sqrt(2)
        return dBR / 2.
    else:
        return dBR

def bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, l1, l2, epsrel=0.005):
    scale = config['renormalization scale']['bpll']
    label = meson_quark[(B,P)] + l1 + l2 # e.g. bsmumu, bdtautau
    wc = wc_obj.get_wc(label, scale, par)
    if all([abs(v) < 1e-12 for v in wc.values()]):
        # if all WCs are essentially zero, return BR=0
        return 0
    def obs(q2):
        return bpll_dbrdq2(q2, wc, par, B, P, l1, l2)
    return flavio.math.integrate.nintegrate(obs, q2min, q2max, epsrel=epsrel)/(q2max-q2min)

# Functions returning functions needed for Prediction instances


def bpll_dbrdq2_tot_func(B, P, l1, l2):
    def fct(wc_obj, par):
        mB = par['m_'+B]
        mP = par['m_'+P]
        ml1 = par['m_'+l1]
        ml2 = par['m_'+l2]
        q2max = (mB-mP)**2
        q2min = (ml1+ml2)**2
        return bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, l1, l2)*(q2max-q2min)
    return fct

def bpll_dbrdq2_tot_lfv_comb_func(B, P, l1, l2):
    def fct(wc_obj, par):
        mB = par['m_'+B]
        mP = par['m_'+P]
        ml1 = par['m_'+l1]
        ml2 = par['m_'+l2]
        q2max = (mB-mP)**2
        q2min = (ml1+ml2)**2
        return (
            + bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, l1, l2)
            + bpll_dbrdq2_int(q2min, q2max, wc_obj, par, B, P, l2, l1)
        )*(q2max-q2min)
    return fct


# Observable and Prediction instances
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


# Lepton flavour violating decays
def _define_obs_B_Mll(M, ll):
    _process_tex = _hadr_lfv[M]['tex']+' '+_tex_lfv[''.join(ll)]
    _process_taxonomy = r'Process :: $b$ hadron decays :: FCNC decays :: $B\to P\ell^+\ell^-$ :: $' + _process_tex + r"$"
    _obs_name = "BR("+M+''.join(ll)+")"
    _obs = Observable(_obs_name)
    _obs.set_description(r"Total branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy)
    return _obs_name

for M in _hadr_lfv:
    for ll in [('e','mu'), ('mu','e'), ('e','tau'), ('tau','e'), ('mu','tau'), ('tau','mu')]:
        _obs_name = _define_obs_B_Mll(M, ll)
        Prediction(_obs_name, bpll_dbrdq2_tot_func(_hadr_lfv[M]['B'], _hadr_lfv[M]['P'], ll[0], ll[1]))
    for ll in [('e','mu'), ('e','tau'), ('mu','tau')]:
        # Combined l1+ l2- + l2+ l1- lepton flavour violating decays
        _obs_name = _define_obs_B_Mll(M, ('{0}{1},{1}{0}'.format(*ll),))
        Prediction(_obs_name, bpll_dbrdq2_tot_lfv_comb_func(_hadr_lfv[M]['B'], _hadr_lfv[M]['P'], ll[0], ll[1]))
