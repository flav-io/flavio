import flavio
from math import pi,log

"""Functions for imroved Isgur-Wise relations to obtain tensor
form factor f_T from f_+ and f_0 in the heavy quark limit."""


def improved_isgur_wise(q2, ff, par, B, P, scale):
    mB = par['m_'+B]
    mP = par['m_'+P]
    mb = flavio.physics.running.running.get_mb(par, scale)
    alpha_s = flavio.physics.running.running.get_alpha(par, scale)['alpha_s']
    # eq. (3.8) of arXiv:1006.5013
    kappa = 1-2*alpha_s/(3*pi)*log(scale/mb)
    # eq. (3.6) of arXiv:1006.5013
    ff['fT'] = kappa*ff['f+']*mB*(mB+mP)/q2
    return ff
