r"""Functions for $e^+ e^-\to l^+ l^- of various flavours"""
# Written by Ben Allanach 

import flavio
import numpy as np

# some functions for use by ee_ee function
def usq_o_ssq(x):
    return 0.25 * (x**3 / 3. + x**2 + x)

def usq_o_st (x):
    return 0.5 * (x + 2 * np.log(1 - x))

def tsq_o_ssq(x):
    return 0.25 * (x - x**2 + x**3 / 3.)        

def s_o_t(x):
    return 2 * np.log(1. - x)

def s_o_t_mz(x, mzsq_o_s):
    return 2 * np.log(1. - x + 2 * mzsq_o_s)

def usq_o_s_o_mz(x, mzsq_o_s):
    return 0.5 * (0.5 * (x + 1.) * (5 + 4 * mzsq_o_s + x) +
                  (2 + 2 * mzsq_o_s)**2 * np.log(1 + 2 * mzsq_o_s - x))
    
# predicted difference to SM correction from SMEFT operators of e+e-->mumu for LEP2 energy E and family fam total cross-section. cthmin/max are the minimum and maximum cosines of the scattering angle in the current bin
def ee_ee(C, par, E, cthmin, cthmax):
    # Check energy E is correct
    if (E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and
        E != 199.5 and E != 201.6 and E != 204.9 and E != 206.6):
        raise ValueError('ee_ee called with incorrect LEP2 energy {} GeV.'.format(E))
    #    
    # For now, delta g couplings have been NEGLECTED
    PI = 3.141592653589793
    s = E * E
    mz = par['m_Z']
    GF = par['GF']
    alpha = par['alpha_e']
    s2w   = par['s2w']
    mzsq_o_s = mz**2 / s
    gzeL  = -0.5 + s2w
    gzeR  = s2w
    eSq   = 4 * PI * alpha
    gLsq  = eSq / s2w
    gYsq  = gLsq * s2w / (1. - s2w)
    vSq   = 1. / (np.sqrt(2.) * GF)
    # The fi are functions of cthmin and cthmax after integration
    f1 = usq_o_ssq(cthmax) - usq_o_ssq(cthmin) + usq_o_st(cthmax) - usq_o_st(cthmin)
    f2 = (usq_o_ssq(cthmax) - usq_o_ssq(cthmin)) / (1. - mz**2 / s)
    f3 = usq_o_s_o_mz(cthmax, mzsq_o_s) - usq_o_s_o_mz(cthmin, mzsq_o_s)
    f4 = tsq_o_ssq(cthmax) - tsq_o_ssq(cthmin)
    f5 = s_o_t(cthmax) - s_o_t(cthmin)
    f6 = (tsq_o_ssq(cthmax) - tsq_o_ssq(cthmin)) / (1. - mz**2 / s)
    f7 = s_o_t_mz(cthmax, mzsq_o_s) - s_o_t_mz(cthmin, mzsq_o_s)
    # I've integrated the costheta's beforehand, but it all needs checking!
    # Expression from 1511.07434v2: the factors of 2 come from difference
    # between Wilson convention and Falkowski's for symmetry factor of WCs
    res   = 1. / (8 * PI) * (
        eSq * (C['ll_1111'] + C['ee_1111']) * f1 * 2 +
        (gLsq + gYsq) * (gzeL**2 * C['ll_1111'] + gzeR**2 *  C['ee_1111'])
        * (f2 + f3)  * 2+
        C['le_1111'] * (
            eSq * (f4 + f5) +
            (gLsq + gYsq) * gzeL * gzeR  * (f6 + f7)
        )
    )
    # The following numerical check made it looks like the constants have the correct values, meaning the conventions are understood: 21/2/23
    # print('# DEBUG: MZ=', np.sqrt(vSq * (gLsq + gYsq) / 4.),' MW=', np.sqrt(gLsq * vSq / 4.),' PI=',PI, ' v=',np.sqrt(vSq))
    conversion_factor = 0.389397e9 / (cthmax - cthmin) # To convert cross-sections in GeV^(-2) to pb
    return res * conversion_factor

def ee_ee_obs(wc_obj, par, E,cthmin, cthmax):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_ee(C, par, E, cthmin, cthmax)

_process_tex = r"e^+e^- \to e^+e^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to e^+e^-$ :: $' + _process_tex + r"$"

_obs_name = "dsigma(ee->ee)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E', 'cthmin', 'cthmax']
flavio.classes.Prediction(_obs_name, ee_ee_obs)
_obs.set_description(r"Differential cross section of $" + _process_tex + r"$ at energy $E$ minus that of the SM")
_obs.tex = r"$d\sigma(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)


