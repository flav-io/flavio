r"""Functions for $e^+ e^-\to l^+ l^- of various flavours"""
# Written by Ben Allanach

from IPython.display import display
from math import pi, sqrt
from flavio.physics.zdecays.smeftew import gV_SM, gA_SM, _QN
from flavio.physics.common import add_dict
from flavio.classes import Observable, Prediction
from flavio.physics import ckm as ckm_flavio
import numpy as np
import flavio.physics.zdecays.smeftew as smeftew
import flavio

# Kronecker delta
def delta(a, b):
    return int(a == b)

# predicted ratio to SM with SMEFT operators of e+e-->mu+mu- or tau+tau- for LEP2 energy E and family fam=2,3 total cross-section. afb should be true for forward-backward asymmetry, whereas it should be false for the total cross-section. Programmed by BCA 22/2/23, checked and corrected 27/2/23
def ee_ll(C, par, E, fam):
    # Check energy E is correct
    if (E != 136.3 and E != 161.3 and E != 172.1 and E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.8 and E!= 204.8 and E!= 206.5):
        raise ValueError('ee_ll called with incorrect LEP2 energy {} GeV.'.format(E))
    s = E * E
    mz = par['m_Z']
    gammaZ = 1 / par['tau_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * pi * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    g_cw  = sqrt(gLsq + gYsq)
    vSq   = 1. / (np.sqrt(2.) * GF)
    res   = 0
    fac   = 1
    div   = 24
    # Need to define complex matrix element first, then take mod squared and multiply by the form factor of each piece.
    sigma_tot_NP = 0
    sigma_tot_SM = 0
    if (fam != 2 and fam != 3):
        raise ValueError(f'ee_ll called with incorrect family {fam} - should be 2 or 3')
    gVe  = g_cw * gV_SM('e', par) 
    gAe  = g_cw * gA_SM('e', par)
    l_type = 'none'
    if (fam == 2):
        l_type = 'mu'
    elif (fam == 3):
        l_type = 'tau'
    gVei = g_cw * gV_SM(l_type, par)
    gAei = g_cw * gA_SM(l_type, par)
    # My ordering for X and Y is (1, 2):=(L, R). 
    for X in range(1, 2):
        for Y in range(1 ,2):
            gexSM = (gVe + gAe) * delta(X, 2) + (gVe - gAe) * delta(X, 1)
            geySM = (gVei + gAei) * delta(Y, 2) + (gVei - gAei) * delta(Y, 1)
            gex   = gexSM + (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(X, 1)
            gey   = geySM + (smeftew.d_gVl(l_type, l_type, par, C) + smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 2) + (smeftew.d_gVl(l_type, l_type, par, C) - smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 1)
            NSM = eSq / s + gexSM * geySM / (s - mz**2 + 1j * mz * gammaZ)
            N = eSq / s + gex * gey / (s - mz**2 + 1j * mz * gammaZ) + (C[f'll_11{fam}{fam}'] + C[f'll_1{fam}{fam}1']) * delta(X, 1) * delta(Y, 1) + C[f'ee_11{fam}{fam}'] * delta(X, 2) * delta(Y, 2) + C[f'le_11{fam}{fam}'] * delta(X, 1) * delta(Y, 2) + C[f'le_{fam}{fam}11'] * delta(X, 2) * delta(Y, 1)
            sigma_tot_NP += abs(N)**2
            sigma_tot_SM += abs(NSM)**2
    # This next contribution has no helicity structure in SM and therefore doesn't interfere with it
    sigma_tot_NP += 3 * abs(C[f'le_{fam}{fam}11'])**2
    return (sigma_tot_NP / sigma_tot_SM)

def ee_ll_obs(wc_obj, par, E, fam):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll(C, par, E, fam)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "R_sigma(ee->ll)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'fam']
Prediction(_obs_name, ee_ll_obs)
_obs.set_description(r"Ratio of cross section of $" + _process_tex + r"$ at energy $E$ to that of the SM")
_obs.tex = r"$R_\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)

# predicted ratio of AFB to SM from SMEFT operators of e+e-->mumu/tautau for LEP2 energy E and family fam total cross-section. AfbSM is the SM prediction for AFB. fam=2 or 3 is the lepton family. Programmed by BCA 27/2/23
def ee_ll_afb(C, par, E, fam):
    # Check energy E is correct
    # Check energy E is correct
    if (E != 136.3 and E != 161.3 and E != 172.1 and E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.8 and E!= 204.8 and E!= 206.5):
        raise ValueError('ee_ll_afb called with incorrect LEP2 energy {} GeV.'.format(E))
    s = E * E
    mz = par['m_Z']
    gammaZ = 1 / par['tau_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * pi * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    g_cw  = sqrt(gLsq + gYsq)
    vSq   = 1. / (np.sqrt(2.) * GF)
    res   = 0
    fac   = 1
    div   = 64
    # Need to define complex matrix element first, then take mod squared and multiply by the form factor of each piece.
    afb_tot    = 0
    afb_tot_SM = 0
    if (fam != 2 and fam != 3):
        raise ValueError(f'ee_ll called with incorrect family {fam} - should be 2 or 3')
    gVe  = g_cw * gV_SM('e', par) 
    gAe  = g_cw * gA_SM('e', par)
    l_type = 'none'
    if (fam == 2):
        l_type = 'mu'
    elif (fam == 3):
        l_type = 'tau'
    gVei = g_cw * gV_SM(l_type, par)
    gAei = g_cw * gA_SM(l_type, par)
    for X in range(1, 2):
        for Y in range(1 ,2):
            gexSM = (gVe + gAe) * delta(X, 2) + (gVe - gAe) * delta(X, 1)
            geySM = (gVei + gAei) * delta(Y, 2) + (gVei - gAei) * delta(Y, 1)
            gex   = gexSM + (smeftew.d_gVl('e', 'e', par, C) + smeftew.d_gAl('e', 'e', par, C)) * delta(X, 2) + (smeftew.d_gVl('e', 'e', par, C) - smeftew.d_gAl('e', 'e', par, C)) * delta(X, 1)
            gey   = geySM + (smeftew.d_gVl(l_type, l_type, par, C) + smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 2) + (smeftew.d_gVl(l_type, l_type, par, C) - smeftew.d_gAl(l_type, l_type, par, C)) * delta(Y, 1)
            # Everything below needs redoing
            NSM = eSq / s + gexSM * geySM / (s - mz**2 + 1j * mz * gammaZ)
            N = eSq / s + gex * gey / (s - mz**2 + 1j * mz * gammaZ) + (C[f'll_11{fam}{fam}'] + C[f'll_1{fam}{fam}1']) * delta(X, 1) * delta(Y, 1) + C[f'ee_11{fam}{fam}'] * delta(X, 2) * delta(Y, 2) + C[f'le_11{fam}{fam}'] * delta(X, 1) * delta(Y, 2) + C[f'le_{fam}{fam}11'] * delta(X, 2) * delta(Y, 1)
            afb_tot += s / (64 * pi) * abs(N)**2 * (2 * delta(X, Y) - 1)
            afb_tot_SM += s / (64 * pi) * abs(NSM)**2 * (2 * delta(X, Y) - 1)
    return afb_tot / afb_tot_SM

def ee_ll_afb_obs(wc_obj, par, E, fam):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll_afb(C, par, E, fam)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "R_Afb(ee->ll)"
_obs = Observable(_obs_name)
_obs.arguments = ['E', 'fam']
Prediction(_obs_name, ee_ll_afb_obs)
_obs.set_description(r"$A_{FB}/A_{FB}(SM) of $" + _process_tex + r"$ at energy $E$")
_obs.tex = r"$A_{FB}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
