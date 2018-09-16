r"""Functions for $\tau\to P\ell$."""

import flavio
from math import sqrt, pi
from flavio.physics.common import lambda_K

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
    mlep = par['m_' + lep]
    fpi0 = 0.130 # FIX me 
    F = {}
    for q in 'ud':
        F['LL' + q] = wc['CVLL_tau{}{}'.format(lep, 2 * q)]  # FLL
        F['RR' + q] = wc['CVRR_tau{}{}'.format(lep, 2 * q)]  # FRR
        F['LR' + q] = wc['CVLR_tau{}{}'.format(lep, 2 * q)]  # FLR
        F['RL' + q] = wc['CVLR_{}tau{}'.format(2 * q, lep)]  # FLRqq
    if P == 'pi0':
        gL = fpi0*((F['LRu'] - F['LRd']) / 2 - (F['LLu'] - F['LLd']) / 2)/2
        gR = fpi0*((F['RRu'] - F['RRd']) / 2 - (F['RLu'] - F['RLd']) / 2)/2       
    norm = 1 / (16 * pi* mtau**3 )
    MEsq = mP**2/2 *( (abs(gL)**2 + abs(gR)**2)*( (mlep**2- mtau**2)**2/mP**2 -mlep**2 -mtau**2 )  
                        + 4*mlep*mtau*(gL.imag * gR.real + gL.real * gR.imag) )  \
         * lambda_K(mtau**2,mlep**2, mP**2 )
#   Method 1 : new result
    brtaupil = norm * MEsq * par['tau_tau']

#   Method 2 : Eq. 29 of hep-ph/0404211
    if P == 'pi0':
        FL = -(F['LRu'] - F['LRd']) / 2 + (F['LLu'] - F['LLd']) / 2
        FR = -(F['RRu'] - F['RRd']) / 2 + (F['LRu'] - F['LRd']) / 2
    full = 1/4/GF**2/0.95* (abs(FL)**2 + abs(FR)**2) * 0.11

    return full#brtaupil


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
