r"""Functions for $K^0\to \ell^+\ell^-$ decays"""

import flavio
from flavio.config import config
from flavio.physics.bdecays.common import lambda_K
from flavio.physics import ckm
from flavio.physics.kdecays.wilsoncoefficients import wilsoncoefficients_sm_sl
from flavio.physics.common import add_dict
from math import pi, sqrt


def amplitudes(par, wc, l1, l2):
    r"""Amplitudes P and S entering the $K\to\ell_1^+\ell_2^-$ observables.

    Parameters
    ----------

    - `par`: parameter dictionary
    - `wc`: Wilson coefficient dictionary
    - `K`: should be `'KL'` or `'KS'`
    - `l1` and `l2`: should be `'e'` or `'mu'`
    """
    # masses
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mK = par['m_K0']
    # Wilson coefficients
    qqll = 'sd' + l1 + l2
    # For LFV expressions see arXiv:1602.00881 eq. (5)
    C9m = wc['C9_'+qqll] - wc['C9p_'+qqll]  # only relevant for l1 != l2
    C10m = wc['C10_'+qqll] - wc['C10p_'+qqll]
    CPm = wc['CP_'+qqll] - wc['CPp_'+qqll]
    CSm = wc['CS_'+qqll] - wc['CSp_'+qqll]
    P = (ml2 + ml1)/mK * C10m + mK * CPm  # neglecting mu, md
    S = (ml2 - ml1)/mK * C9m  + mK * CSm  # neglecting mu, md
    xi_t = ckm.xi('t', 'sd')(par)
    return xi_t * P, xi_t * S


def amplitudes_LD(par, K, l):
    r"""Long-distance amplitudes entering the $K\to\ell^+\ell^-$ observables."""
    ml = par['m_' + l]
    mK = par['m_' + K]
    s2w = par['s2w']
    pre = 2 * ml / mK / s2w
    # numbers extracted from arXiv:1711.11030
    flavio.citations.register("Chobanova:2017rkj")
    ASgaga = 2.49e-4 * (-2.821 + 1.216j)
    ALgaga = 2.02e-4 * (par['chi_disp(KL->gammagamma)'] - 5.21j)
    S = pre * ASgaga
    P = -pre * ALgaga
    return S, P


def amplitudes_eff(par, wc, K, l1, l2, ld=True):
    r"""Effective amplitudes entering the $K\to\ell_1^+\ell_2^-$ observables."""
    P, S = amplitudes(par, wc, l1, l2)
    if l1 != l2 or not ld:
        SLD = 0
        PLD = 0
    else:
        SLD, PLD = amplitudes_LD(par, K, l1)
    if K == 'KS' and l1 == l2:
        Peff = P.imag
        Seff = S.real + SLD
    if K == 'KL':
        Peff = P.real + PLD
        Seff = S.imag
    return Peff, Seff


def get_wc(wc_obj, par, l1, l2):
    scale = config['renormalization scale']['kdecays']
    label = 'sd' + l1 + l2
    wcnp = wc_obj.get_wc(label, scale, par)
    if l1 == l2:
        # include SM contributions for LF conserving decay
        _c = wilsoncoefficients_sm_sl(par, scale)
        xi_t = ckm.xi('t', 'sd')(par)
        xi_c = ckm.xi('c', 'sd')(par)
        wcsm = {'C10_sd' + l1 + l2: _c['C10_t'] + xi_c / xi_t * _c['C10_c']}
    else:
        wcsm = {}

    return add_dict((wcsm, wcnp))


def br_kll(par, wc_obj, K, l1, l2, ld=True):
    r"""Branching ratio of $K\to\ell_1^+\ell_2^-$"""
    # parameters
    wc = get_wc(wc_obj, par, l1, l2)
    GF = par['GF']
    alphaem = par['alpha_e']
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mK = par['m_K0']
    tauK = par['tau_'+K]
    fK = par['f_K0']
    # appropriate CKM elements
    N = 4 * GF / sqrt(2) * alphaem / (4 * pi)
    beta = sqrt(lambda_K(mK**2, ml1**2, ml2**2)) / mK**2
    beta_p = sqrt(1 - (ml1 + ml2)**2 / mK**2)
    beta_m = sqrt(1 - (ml1 - ml2)**2 / mK**2)
    prefactor = 2 * abs(N)**2 / 32. / pi * mK**3 * tauK * beta * fK**2
    Peff, Seff = amplitudes_eff(par, wc, K, l1, l2, ld=ld)
    return prefactor * (beta_m**2 * abs(Peff)**2 + beta_p**2 * abs(Seff)**2)


# function returning function needed for prediction instance
def br_kll_fct(K, l1, l2):
    def f(wc_obj, par):
        return br_kll(par, wc_obj, K, l1, l2)
    return f

def br_kll_fct_lsum(K, l1, l2):
    def f(wc_obj, par):
        return br_kll(par, wc_obj, K, l1, l2) + br_kll(par, wc_obj, K, l2, l1)
    return f


_tex = {'e': 'e', 'mu': r'\mu'}
_tex_p = {'KL': r'K_L', 'KS': r'K_S',}

for l in ['e', 'mu']:
    for P in _tex_p:
        _obs_name = "BR({}->{}{})".format(P, l, l)
        _obs = flavio.classes.Observable(_obs_name)
        _process_tex = _tex_p[P] + r"\to "+_tex[l]+r"^+"+_tex[l]+r"^-"
        _process_taxonomy = r'Process :: $s$ hadron decays :: FCNC decays :: $K\to \ell\ell$ :: $' + _process_tex + r'$'
        _obs.add_taxonomy(_process_taxonomy)
        _obs.set_description(r"Branching ratio of $" + _process_tex +r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        flavio.classes.Prediction(_obs_name, br_kll_fct(P, l, l))


# LFV decay
for P in _tex_p:
    _obs_name = "BR({}->emu,mue)".format(P)
    _obs = flavio.classes.Observable(_obs_name)
    _process_tex = _tex_p[P] + r"\to e^\pm\mu^\mp"
    _process_taxonomy = r'Process :: $s$ hadron decays :: FCNC decays :: $K\to \ell\ell$ :: $' + _process_tex + r'$'
    _obs.add_taxonomy(_process_taxonomy)
    _obs.set_description(r"Branching ratio of $" + _process_tex +r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    flavio.classes.Prediction(_obs_name, br_kll_fct_lsum(P, 'e', 'mu'))
