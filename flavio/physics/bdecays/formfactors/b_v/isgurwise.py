import flavio
from math import pi,log

"""Functions for imroved Isgur-Wise relations to obtain tensor
form factors from vector form factors in the heavy quark limit."""


def improved_isgur_wise(q2, ff, par, B, V, scale):
    mB = par['m_'+B]
    mV = par['m_'+V]
    mb = flavio.physics.running.running.get_mb(par, scale)
    alpha_s = flavio.physics.running.running.get_alpha(par, scale)['alpha_s']
    D0v = 0
    C0v = 0
    # eq. (3.8) of arXiv:1006.5013
    kappa = 1-2*alpha_s/(3*pi)*log(scale/mb)
    # eq. (3.6) of arXiv:1006.5013
    ff['T1'] = kappa*ff['V']
    ff['T2'] = kappa*ff['A1']
    # converting A12 -> A2
    A2 = (((mB + mV)*(-16*ff['A12']*mB*mV**2
        + ff['A1']*(mB + mV)*(mB**2 - mV**2 - q2)))
        / (mB**4 + (mV**2 - q2)**2 - 2*mB**2*(mV**2 + q2)))
    T3 = kappa*A2*mB**2/q2
    # converting T3->T23
    ff['T3'] = T3
    ff['A2'] = A2
    ff['T23'] = (((mB - mV)*(mB + mV)* (mB**2 + 3*mV**2 - q2)*ff['T2']
            - ((mB - mV)**2 - q2)* ((mB + mV)**2 - q2)*T3)/ (8.*mB*(mB - mV)*mV**2))
    return ff
