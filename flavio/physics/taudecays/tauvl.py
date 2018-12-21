r"""Functions for $\tau\to V\ell$."""

import flavio
from flavio.physics.taudecays import common
from math import sqrt, pi
import numpy as np


# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }

def get_wcs(wc, q, lep):
        return np.array([
            wc['CVLL_tau{}{}'.format(lep, 2 * q)],
            wc['CVLR_tau{}{}'.format(lep, 2 * q)],
            wc['CVLR_{}tau{}'.format(2 * q, lep)],
            wc['CVRR_tau{}{}'.format(lep, 2 * q)],
            wc['CTRR_{}tau{}'.format(lep, 2 * q)],
            wc['CTRR_tau{}{}'.format(lep, 2 * q)],
        ])

def br_tauvl(wc_obj, par, V, lep):
    r"""Branching ratio of $\tau^+\to V^0\ell^+$."""
    scale = flavio.config['renormalization scale']['taudecays']
    sec = wcxf_sector_names['tau', lep]
    wc = wc_obj.get_wc(sec, scale, par, nf_out=4)
    alpha = flavio.physics.running.running.get_alpha_e(par, scale, nf_out=3)
    e = sqrt(4 * pi * alpha)
    mtau = par['m_tau']
    ml = par['m_' + lep]
    mV = par['m_' + V]
    fV = par['f_' + V]
    fTV = flavio.physics.running.running.get_f_perp(par, V, scale)
    Cgamma_taul = wc['Cgamma_tau{}'.format(lep)]
    Cgamma_ltau = wc['Cgamma_{}tau'.format(lep)]
    if V == 'rho0':
        g_u = get_wcs(wc, 'u', lep)
        g_d = get_wcs(wc, 'd', lep)
        g = (g_u-g_d)/sqrt(2)
        KV = -1/sqrt(2)*e
    if V == 'phi':
        g = get_wcs(wc, 's', lep)
        KV = 1/3*e
    gL = mV*fV/2 * (g[0] + g[1])
    gR = mV*fV/2 * (g[2] + g[3])
    gTL  = +fTV * g[4].conjugate() + 2*fV*KV/mV * Cgamma_ltau.conjugate()
    gtTL = -fTV * g[4].conjugate()
    gTR  = +fTV * g[5] + 2*fV*KV/mV * Cgamma_taul
    gtTR = +fTV * g[5]
    return (par['tau_tau']
            * common.GammaFvf(mtau, mV, ml, gL, gR, gTL, gtTL, gTR, gtTR) )


# function returning function needed for prediction instance
def br_tauvl_fct(V, lep):
    def f(wc_obj, par):
        return br_tauvl(wc_obj, par, V, lep)
    return f

# Observable and Prediction instances

_had = {'rho0': r'\rho^0', 'phi': r'\phi'}
_shortname = {'rho0': 'rho', 'phi': 'phi'}
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
