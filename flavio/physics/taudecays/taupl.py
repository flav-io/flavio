r"""Functions for $\tau\to P\ell$."""

import flavio
from math import sqrt, pi
from flavio.physics.common import lambda_K
from flavio.physics.taudecays import common
from flavio.physics import ckm
from flavio.physics.running import running

# names of LFV sectors in WCxf
wcxf_sector_names = {('tau', 'mu'): 'mutau',
                     ('tau', 'e'): 'taue',
                     ('mu', 'e'): 'mue', }

def br_taupl(wc_obj, par, P, lep):
    r"""Branching ratio of $\tau^+\to p^0\ell^+$."""
    scale = flavio.config['renormalization scale']['taudecays']
    GF = par['GF']
    alphaem = running.get_alpha(par, 4.8)['alpha_e']
    mP = par['m_' + P]
    mtau = par['m_tau']
    mu = par['m_u']
    md = par['m_d']
    mlep = par['m_' + lep]
    fP = par['f_' + P]  

    if P == 'pi0':
        sec = wcxf_sector_names['tau', lep]
        wc = wc_obj.get_wc(sec, scale, par, nf_out=4)
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

        vL = fP*((F['LRu'] - F['LLu'])/2  - (F['LRd'] - F['LLd'])/2 )/sqrt(2)
        vR = fP*((F['RRu'] - F['RLu'])/2  - (F['RRd'] - F['RLd'])/2 )/sqrt(2)
        sL = fP*mP**2/(mu+md)*((S['LRu']-S['LLu'])/2 - (S['LRd']-S['LLd'])/2)/sqrt(2)
        sR = fP*mP**2/(mu+md)*((S['RRu']-S['RLu'])/2 - (S['RRd']-S['RLd'])/2)/sqrt(2) 
        gL = sL+ (-vL*mlep + vR*mtau)
        gR = sR+ (-vR*mlep + vL*mtau)

    elif P == 'K0':
       xi_t = ckm.xi('t','ds')(par)
       qqll = 'sdtau' + lep
       wc = wc_obj.get_wc(qqll, scale, par, nf_out=4)
       mq1 = flavio.physics.running.running.get_md(par, scale)
       mq2 = flavio.physics.running.running.get_ms(par, scale)

       C9m =  -wc['C9_'+qqll]  + wc['C9p_'+qqll]
       C10m = -wc['C10_'+qqll] + wc['C10p_'+qqll]
       CSm =   wc['CS_'+qqll]  - wc['CSp_'+qqll]
       CPm =   wc['CP_'+qqll]  - wc['CPp_'+qqll]
       Kv = xi_t * 4*GF/sqrt(2) * alphaem/(4*pi)
       Ks = Kv * mq2
       gV = - fP* Kv * C9m/2
       gA = - fP* Kv * C10m/2
       gS = - fP* Ks * mP**2 * CSm/2 / (mq1+mq2)
       gP = - fP* Ks * mP**2 * CPm/2 / (mq1+mq2)
       gVS = gS + gV*(mtau -mlep)
       gAP = gP - gA*(mtau +mlep)
       gL =  gVS - gAP
       gR =  gVS + gAP
    return par['tau_tau'] * common.GammaFsf(mtau, mP, mlep, gL, gR)

# function returning function needed for prediction instance
def br_taupl_fct(P, lep):
    def f(wc_obj, par):
        return br_taupl(wc_obj, par, P, lep)
    return f

# Observable and Prediction instances

_had = {'pi0': r'\pi^0', 'K0' : r'K^0'}
_shortname = {'pi0': 'pi', 'K0': 'K'}
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
