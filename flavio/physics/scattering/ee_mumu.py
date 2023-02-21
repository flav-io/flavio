r"""Functions for $e^+ e^-\to l^+ l^-"""

import flavio
import numpy as np

# predicted difference to SM correction from SMEFT operators of e+e-->mumu for LEP2 energy E
def ee_mumu(C, par, E):
    # Check energy E is correct
    if (E != 182.7 and E != 188.6 and E != 191.6 and E != 195.5 and E != 199.5 and E!= 201.6 and E!= 206.6):
        raise ValueError('In lep2mumu: called with incorrect LEP2 energy {} GeV.'.format(E))
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
    # From 
    res   = 1. / (24. * PI * vSq) * (
        eSq * (C['ll_1122'] + C['ll_1221'] + C['ee_1122'] +
               C['le_1122'] + C['le_2211']) +
        s * (gLsq + gYsq) / (s - mz**2) * (
        gzeL**2 * (C['ll_1122'] + C['ll_1221']) +
        gzeR**2 *  C['ee_1122'] +
        gzeL * gzeR * (C['le_1122'] + C['le_2211'])
        )
    )
    # The following numerical check made it looks like the constants are pretty much correct: 21/2/23
    # print('# DEBUG: MZ=', np.sqrt(vSq * (gLsq + gYsq) / 4.),' MW=', np.sqrt(gLsq * vSq / 4.),' PI=',PI, ' v=',np.sqrt(vSq))
    return res 

def ee_mumu_obs(wc_obj, par, E):
    scale = flavio.config['renormalization scale']['ee_ww'] # Use LEP2 renorm scale
    C = wc_obj.get_wcxf(sector='all', scale=scale, par=par,
                        eft='SMEFT', basis='Warsaw')
    return ee_mumu(C, par, E)

_process_tex = r"e^+e^- \to \mu^+\mu^-"
_process_taxonomy = r'Process :: $e^+e^-$ scattering :: $e^+e^-\to ll$ :: $' + _process_tex + r"$"

_obs_name = "dsigma(ee->mumu)"
_obs = flavio.classes.Observable(_obs_name)
_obs.arguments = ['E']
flavio.classes.Prediction(_obs_name, ee_mumu_obs)
_obs.set_description(r"Cross section of $" + _process_tex + r"$ at energy $E$ minus that of the SM")
_obs.tex = r"$R(" + _process_tex + r")$"
_obs.add_taxonomy(_process_taxonomy)
