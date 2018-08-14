r"""Functions for $\tau\to V\ell$."""

import flavio
from math import sqrt, pi


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def br_tauvl(wc_obj, par, V, lep):
    r"""Branching ratio of $\tau^+\to V^0\ell^+$."""
    scale = flavio.config['renormalization scale']['taudecays']
    sec = wcxf_sector_names['tau', lep]
    wc = wc_obj.get_wc(sec, scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=3)
    e = sqrt(4 * pi * alpha)
    mV = par['m_' + V]
    mtau = par['m_tau']
    x = mV**2 / mtau**2
    pre_wc_1 = 1 / e / mtau
    e2DLg = e**2 * pre_wc_1 * wc['Cgamma_tau{}'.format(lep)]
    e2DRg = e**2 * pre_wc_1 * wc['Cgamma_{}tau'.format(lep)].conjugate()
    F = {}
    for q in 'ud':
        F['LL' + q] = wc['CVLL_tau{}{}'.format(lep, 2 * q   )]
        F['RR' + q] = wc['CVRR_tau{}{}'.format(lep, 2 * q)]
        F['LR' + q] = wc['CVLR_tau{}{}'.format(lep, 2 * q)]
        F['RL' + q] = wc['CVLR_{}tau{}'.format(2 * q, lep)]
    if V == 'rho0':
        FL = (F['LLu'] - F['LLd']) / 2 + (F['LRu'] - F['LRd']) / 2
        FR = (F['RLu'] - F['RLd']) / 2 + (F['RRu'] - F['RRd']) / 2
        Vud = flavio.physics.ckm.get_ckm(par)[0, 0]
        norm = 1 / (4 * par['GF']**2 * abs(Vud)**2) * par['BR(tau->rhonu)']
    rWC = (abs(FL)**2 + abs(FR)**2
           - 6 / (1 + 2 * x) * (e2DLg * FL.conjugate() + e2DRg * FR.conjugate()).real
           + (2 + x) / (x * (1 + 2 * x)) * (abs(e2DLg)**2 + abs(e2DRg)**2))
    return norm * rWC


# function returning function needed for prediction instance
def br_tauvl_fct(V, lep):
    def f(wc_obj, par):
        return br_tauvl(wc_obj, par, V, lep)
    return f

# Observable and Prediction instances

_had = {'rho0': r'\rho^0',}
_shortname = {'rho0': 'rho',}
_lep = {'e': ' e', 'mu': r'\mu',}

for V in _had:
    for l in _lep:
        _obs_name = "BR(tau->" + _shortname[V] + l + r")"
        _obs = flavio.classes.Observable(_obs_name)
        _process_tex = r"\tau^+\to " + _had[V] + _lep[l] + r"^+"
        _process_taxonomy = r'Process :: $\tau$ lepton decays :: LFV decays :: $\tau\to V\ell$ :: $' + _process_tex + r"$"
        _obs.add_taxonomy(_process_taxonomy)
        _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        flavio.classes.Prediction(_obs_name, br_tauvl_fct(V, l))
