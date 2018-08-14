"""Functions for Isgur-Wise-like relations to obtain tensor
form factors from vector form factors from the equations of motion
in the heavy quark limit."""

import flavio
from math import pi,log


def isgur_wise(process, q2, ff, par, scale):
    pd = flavio.physics.bdecays.formfactors.b_v.cln.process_dict[process]
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    mb = flavio.physics.running.running.get_mb_pole(par)
    if pd['q'] == 'b->c':
        mq = flavio.physics.running.running.get_mc_pole(par)
    else:
        mq = 0 # neglect m_u,d,s
    # power corrections
    a_T1  = par[process + ' IW a_T1']
    a_T2  = par[process + ' IW a_T2']
    a_T23 = par[process + ' IW a_T23']
    # cf. eq. (11) of arXiv:1503.05534
    ff['T1'] = (mb + mq)/(mB + mV)*ff['V']  * ( 1 + a_T1 )
    ff['T2'] = (mb - mq)/(mB - mV)*ff['A1'] * ( 1 + a_T2 )
    if q2 == 0:
        ff['T23'] = (4*ff['A12']*(mb - mq)*(mB**2 + mV**2))/((mB - mV)**2*(mB + mV))
    else:
        ff['T23'] = ((mb - mq)* (
                ((mB - mV)**2 - q2)* ((mB + mV)**2 - q2)*ff['A0']
                + 8*mB*mV* (-mB**2 + mV**2)* ff['A12']
                ))/ (4.*mB*(mV - mB)*mV*q2) * ( 1 + a_T23 )
    return ff
