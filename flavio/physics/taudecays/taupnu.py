r"""Functions for $\tau\to P\nu$."""

import flavio
from flavio.physics.taudecays import common
from math import pi, sqrt

def br_taupnu(wc_obj, par, P):
    r"""Branching ratio of $\tau^+\to P^+\bar\nu_\ell$."""
    # CKM element
    if P=='pi+':
        Vij = flavio.physics.ckm.get_ckm(par)[0,0] # Vud
        qqlnu = 'dutaunu'
    elif P=='K+':
        Vij = flavio.physics.ckm.get_ckm(par)[0,1] # Vus
        qqlnu = 'sutaunu'
    scale = flavio.config['renormalization scale']['taudecays']
    # Wilson coefficients
    wc = wc_obj.get_wc(qqlnu, scale, par, nf_out=4)
    # add SM contribution to Wilson coefficient
    wc['CV_'+qqlnu] += flavio.physics.bdecays.wilsoncoefficients.get_CVSM(par, scale, nf=4)
    mtau = par['m_tau']
    mP = par['m_'+P]
    rWC = (wc['CV_'+qqlnu] - wc['CVp_'+qqlnu]) + mP**2/mtau * (wc['CS_'+qqlnu] - wc['CSp_'+qqlnu])
    gL = par['GF'] * Vij * rWC * par['f_'+P] * par['m_tau']
    return par['tau_tau'] * common.GammaFsf(mtau, mP, 0, gL, -gL) # gR = -gL for pseudoscalar

# function returning function needed for prediction instance
def br_taupnu_fct(P):
    def f(wc_obj, par):
        return br_taupnu(wc_obj, par, P)
    return f

# Observable and Prediction instances

_had = {'pi+': r'\pi^+', 'K+': r'K^+',}
_shortname = {'pi+': 'pi', 'K+': 'K'}

for P in _had:
    _obs_name = "BR(tau->"+_shortname[P]+"nu)"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $\tau^+\to "+_had[P]+r"\bar\nu$")
    _obs.tex = r"$\text{BR}(\tau^+\to "+_had[P]+r"\bar\nu)$"
    flavio.classes.Prediction(_obs_name, br_taupnu_fct(P))
