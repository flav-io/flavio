r"""Functions for $e^+ e^-\to l^+ l^- of various flavours"""
# Written by Ben Allanach: note ignores small imaginary parts coming from Wilson coefficient running. Not yet appropriate for complex Wilson coefficients.

import flavio
import numpy as np

# predicted difference to SM correction in pb from SMEFT operators of e+e-->mumu for LEP2 energy E and family fam total cross-section. afb should be true for forward-backward asymmetry, whereas it should be false for the total cross-section. Programmed by BCA 22/2/23, checked and corrected 27/2/23
def ee_ll(C, par, E, fam):
    # Check energy E is correct
    if (E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and
        E != 199.5 and E!= 201.6 and E!= 204.9 and E!= 206.6):
        raise ValueError('ee_ll called with incorrect LEP2 energy {} GeV.'.format(E))
        
    # For now, delta g couplings have been NEGLECTED
    PI = 3.141592653589793
    s = E * E
    mz = par['m_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * PI * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    vSq   = 1. / (np.sqrt(2.) * GF)
    res   = 0
    fac   = 1
    div   = 24
    # Expression from 1511.07434v2
    if (fam == 2 or fam == 3):
        res   = 1. / (div * PI) * (
            eSq * np.real_if_close(C[f'll_11{fam}{fam}'].real +
                   C['ll_1' + str(fam) + str(fam) + '1'] +
                   C['ee_11' + str(fam) + str(fam)] +
                   fac * C['le_11' + str(fam) + str(fam)] +
                   fac * C['le_' + str(fam) + str(fam) + '11']) +
            s * (gLsq + gYsq) / (s - mz**2) * np.real_if_close(
                gzeL**2 * (C['ll_11' + str(fam) + str(fam)] +
                           C['ll_1' + str(fam) + str(fam) + '1']) +
                gzeR**2 *  C['ee_11' + str(fam) + str(fam)] +
                fac * gzeL * gzeR * (C['le_11' + str(fam) + str(fam)] +
                               C['le_' + str(fam) + str(fam) + '11'])
            )
        )
    else:
        raise ValueError('ee_ll called with incorrect family {}'.format(fam))
    # The following numerical check made it looks like the constants have the correct values, meaning the conventions are understtod: 21/2/23
    # print('# DEBUG: MZ=', np.sqrt(vSq * (gLsq + gYsq) / 4.),' MW=', np.sqrt(gLsq * vSq / 4.),' PI=',PI, ' v=',np.sqrt(vSq))
    conversion_factor = 0.389397e9 # To convert cross-sections in GeV^(-2) to pb
    return res * conversion_factor

def ee_ll_obs(wc_obj, par, E, fam):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll(C, par, E, fam)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "dsigma(ee->ll)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E', 'fam']
flavio.classes.Prediction(_obs_name, ee_ll_obs)
_obs.set_description(r"Cross section of $" + _process_tex + r"$ at energy $E$ minus that of the SM")
_obs.tex = r"$d\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)


# predicted AFB from SMEFT operators of e+e-->mumu/tautau for LEP2 energy E and family fam total cross-section. AfbSM is the SM prediction for AFB. fam=2 or 3 is the lepton family. Programmed by BCA 27/2/23
def ee_ll_afb(C, par, E, fam, AfbSM, sigmaSM):
    # Check energy E is correct
    if (E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and
        E != 199.5 and E != 201.6 and E != 204.9 and E != 206.6):
        raise ValueError('ee_ll_afb called with incorrect LEP2 energy {} GeV.'.format(E))
        
    # For now, delta g couplings have been NEGLECTED
    PI = 3.141592653589793
    s = E * E
    mz = par['m_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * PI * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    vSq   = 1. / (np.sqrt(2.) * GF)
    res   = 0
    fac   = -1
    div   = 32
    conversion_factor = 0.389397e9 # To convert cross-sections in GeV^(-2) to pb
    # Expression from 1511.07434v2
    if (fam == 2 or fam == 3):
        res   = conversion_factor / (div * PI) * (
            eSq * np.real_if_close(C['ll_11' + str(fam) + str(fam)] +
                   C['ll_1' + str(fam) + str(fam) + '1'] +
                   C['ee_11' + str(fam) + str(fam)] +
                   fac * C['le_11' + str(fam) + str(fam)] +
                   fac * C['le_' + str(fam) + str(fam) + '11']) +
            s * (gLsq + gYsq) / (s - mz**2) * np.real_if_close(
                gzeL**2 * (C['ll_11' + str(fam) + str(fam)] +
                           C['ll_1' + str(fam) + str(fam) + '1']) +
                gzeR**2 *  C['ee_11' + str(fam) + str(fam)] +
                fac * gzeL * gzeR * (C['le_11' + str(fam) + str(fam)] +
                               C['le_' + str(fam) + str(fam) + '11'])
            )
        )
    else:
        raise ValueError('ee_ll_afb called with incorrect family {}'.format(fam))
    # The following numerical check made it looks like the constants have the correct values, meaning the conventions are understtod: 21/2/23
    # print('# DEBUG: MZ=', np.sqrt(vSq * (gLsq + gYsq) / 4.),' MW=', np.sqrt(gLsq * vSq / 4.),' PI=',PI, ' v=',np.sqrt(vSq))
    # New physics contribution to forward-backward cross-section
    dsigma_fb  = res
    # New physics contribution to total cross-section at this energy
    dsigma_tot = ee_ll(C, par, E, fam)
    return AfbSM * (1.0 - dsigma_tot / sigmaSM) + dsigma_fb / sigmaSM

def ee_ll_afb_obs(wc_obj, par, E, fam, AfbSM, sigmaSM):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ll_afb(C, par, E, fam, AfbSM, sigmaSM)

_process_tex = r"e^+e^- \to l^+l^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to l^+l^-$ :: $' + _process_tex + r"$"

_obs_name = "AFB(ee->ll)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E', 'fam', 'AfbSM', 'sigmaSM']
flavio.classes.Prediction(_obs_name, ee_ll_afb_obs)
_obs.set_description(r"$A_{FB} of $" + _process_tex + r"$ at energy $E$")
_obs.tex = r"$A_{FB}(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
