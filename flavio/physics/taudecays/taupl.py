r"""Functions for $\tau\to P\ell$."""

import flavio
from math import sqrt, pi
from flavio.physics.common import lambda_K
from flavio.physics.taudecays import common

# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }


def br_taupl(wc_obj, par, P, lep):
    r"""Branching ratio of $\tau^+\to p^0\ell^+$."""
    scale = flavio.config['renormalization scale']['taudecays']
    sec = wcxf_sector_names['tau', lep]
    wc = wc_obj.get_wc(sec, scale, par, nf_out=4)
    mP = par['m_' + P]
    GF = par['GF']
    mtau = par['m_tau']
    mu = par['m_u']
    md = par['m_d']
    mlep = par['m_' + lep]
    fpi0 = par['f_' + P] #0.130 # FIX me ?
    F = {}
    S = {}
    for q in 'ud':
        F['LL' + q] = wc['CVLL_tau{}{}'.format(lep, 2 * q)]  # FLL
        F['RR' + q] = wc['CVRR_tau{}{}'.format(lep, 2 * q)]  # FRR
        F['LR' + q] = wc['CVLR_tau{}{}'.format(lep, 2 * q)]  # FLR
        F['RL' + q] = wc['CVLR_{}tau{}'.format(2 * q, lep)]  # FLRqq
     
        S['RR' + q] = wc['CSRR_tau{}{}'.format(lep, 2 * q)]  # SRR
        S['RL' + q] = wc['CSRL_tau{}{}'.format(lep, 2 * q)]  # SLR
        S['LL' + q] = wc['CSRR_{}tau{}'.format(lep, 2 * q)].conjugate()  # SRRqq
        S['LR' + q] = wc['CSRL_{}tau{}'.format(lep, 2 * q)].conjugate()  # SLRqq

    if P == 'pi0':
        vL = fpi0*((F['LRu'] - F['LLu'])  + (F['LRd'] - F['LLd']) )/(2*sqrt(2))
        vR = fpi0*((F['RRu'] - F['RLu'])  + (F['RRd'] - F['RLd']) )/(2*sqrt(2))   
        sL =-fpi0*mP**2*((S['LRu']-S['LLu'])/mu + (S['LRd']-S['LLd'])/md)/(4*sqrt(2))
        sR =-fpi0*mP**2*((S['RRu']-S['RLu'])/mu + (S['RRd']-S['RLd'])/md)/(4*sqrt(2)) 
        gL = sL+ (vL*mlep-vR*mtau)
        gR = sR+ (vR*mlep-vL*mtau)
    brtaupil = par['tau_tau'] * common.GammaFsf(mtau, mP, mlep, gL, gR)

#   Method 2 : Eq. 29 of hep-ph/0404211
    if P == 'pi0':
        FL = -(F['LRu'] - F['LRd']) / 2 + (F['LLu'] - F['LLd']) / 2
        FR = -(F['RRu'] - F['RRd']) / 2 + (F['LRu'] - F['LRd']) / 2
    full = 1/4/GF**2/0.95* (abs(FL)**2 + abs(FR)**2) * 0.11

    return brtaupil


# function returning function needed for prediction instance
def br_taupl_fct(P, lep):
    def f(wc_obj, par):
        return br_taupl(wc_obj, par, P, lep)
    return f

# Observable and Prediction instances

_had = {'pi0': r'\pi^0',}
_shortname = {'pi0': 'pi',}
_lep = {'e': ' e', 'mu': r'\mu',}

for P in _had:
    for l in _lep:
        _obs_name = "BR(tau->" + _shortname[P] + l + r")"
        _obs = flavio.classes.Observable(_obs_name)
        _process_tex = r"\tau^+\to " + _had[P] + _lep[l] + r"^+"
        _process_taxonomy = r'Process :: $\tau$ lepton decays :: LFV decays :: $\tau\to P\ell$ :: $' + _process_tex + r"$"
        _obs.add_taxonomy(_process_taxonomy)
        _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        flavio.classes.Prediction(_obs_name, br_taupl_fct(P, l))
