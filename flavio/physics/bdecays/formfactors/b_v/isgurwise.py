"""Functions for imroved Isgur-Wise relations to obtain tensor
form factors from vector form factors in the heavy quark limit."""

import flavio
from math import pi,log


def improved_isgur_wise(process, q2, ff, par, scale):
    pd = flavio.physics.bdecays.formfactors.b_v.cln.process_dict[process]
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    mb = flavio.physics.running.running.get_mb(par, scale)
    alpha_s = flavio.physics.running.running.get_alpha(par, scale)['alpha_s']
    D0v = 0
    C0v = 0
    # eq. (3.8) of arXiv:1006.5013
    kappa = 1-2*alpha_s/(3*pi)*log(scale/mb)
    # power corrections
    a_T1  = par[process + ' IW a_T1']
    a_T2  = par[process + ' IW a_T2']
    a_T23 = par[process + ' IW a_T23']
    # eq. (3.6) of arXiv:1006.5013
    ff['T1'] = kappa*ff['V']  * ( 1 + a_T1 )
    ff['T2'] = kappa*ff['A1'] * ( 1 + a_T2 )
    # derived in analogy to arXiv:1006.5013 using hep-ph/0404250
    ff['T23'] = kappa*ff['A12'] * 2*mB**2/q2 * ( 1 + a_T23 )
    return ff
