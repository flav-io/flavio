r"""Functions for imroved Isgur-Wise relations to obtain tensor
form factor $f_T$ from $f_+$ and $f_0$s in the heavy quark limit."""

import flavio
from math import pi,log



def improved_isgur_wise(process, q2, ff, par, scale):
    pd = flavio.physics.bdecays.formfactors.b_p.bcl.process_dict[process]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    mb = flavio.physics.running.running.get_mb(par, scale)
    alpha_s = flavio.physics.running.running.get_alpha(par, scale)['alpha_s']
    # eq. (3.8) of arXiv:1006.5013
    kappa = 1-2*alpha_s/(3*pi)*log(scale/mb)
    # eq. (3.6) of arXiv:1006.5013
    aT = par[process + ' IW a_T'] # a_T is a relative power correction
    ff['fT'] = kappa*ff['f+']*mB*(mB+mP)/q2 * (1 + aT)
    return ff
