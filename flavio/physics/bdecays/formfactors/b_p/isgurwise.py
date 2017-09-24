r"""Functions for Isgur-Wise-like relation to obtain tensor
form factor $f_T$ from $f_+$ and $f_0$s in the heavy quark limit."""

import flavio
from math import pi,log



def isgur_wise(process, q2, ff, par, scale):
    pd = flavio.physics.bdecays.formfactors.b_p.bcl.process_dict[process]
    mB = par['m_'+pd['B']]
    mP = par['m_'+pd['P']]
    aT = par[process + ' IW a_T'] # a_T is a relative power correction
    ff['fT'] = ff['f+'] * (1 + aT)
    return ff
