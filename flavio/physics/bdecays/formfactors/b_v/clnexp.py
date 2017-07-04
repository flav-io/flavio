from math import sqrt
import numpy as np
from flavio.physics.bdecays.formfactors.b_v.isgurwise import isgur_wise


process_dict = {}
process_dict['B->D*'] =    {'B': 'B0', 'V': 'D*0',   'q': 'b->c'}


def ff(process, q2, par, scale):
    r"""Central value of $B\to V$ form factors in the lattice convention
      CLN parametrization.

    See eqs. (B4)-(B7) of arXiv:1203.2654.
    """
    pd = process_dict[process]
    mB = par['m_'+pd['B']]
    mV = par['m_'+pd['V']]
    w = (mB**2 + mV**2 - q2) / (2*mB*mV)
    z = (sqrt(w+1)-sqrt(2))/(sqrt(w+1)+sqrt(2))
    RV = 2*sqrt(mB*mV)/(mB+mV)
    hA1_1 = par[process + ' CLN h_A1(1)']
    R1_1 = par[process + ' CLN R_1(1)']
    R2_1 = par[process + ' CLN R_2(1)']
    rho2 = par[process + ' CLN rho2']
    # determine R_0(w=1) by requiring that 8mBmV/(mB^2-mV^2)A_12(q^2=0)=A_0(q^2=0)
    w_0 = (mB**2 + mV**2) / (2*mB*mV) # w at q^2=0
    R2_0 = R2_1 + 0.11*(w_0-1) - 0.06*(w_0-1)**2 # R_2 at q^2=0
    R0_0 = (mB + mV - mB * R2_0 + mV * R2_0)/(2 * mV) # R_0 at q^2=0
    R0_1 = R0_0 - (- 0.11*(w_0-1) + 0.01*(w_0-1)**2) # R_0 at w=1 (q^2=max)
    # done determining R_0(w=1)
    hA1 = hA1_1 * (1 - 8*rho2*z + (53*rho2-15)*z**2 - (231*rho2-91)*z**3)
    R1 = R1_1 - 0.12*(w-1) + 0.05*(w-1)**2
    R2 = R2_1 + 0.11*(w-1) - 0.06*(w-1)**2
    R0 = R0_1 - 0.11*(w-1) + 0.01*(w-1)**2
    ff = {}
    ff['A1'] = hA1 * RV * (w+1)/2.
    ff['A0'] = R0/RV * hA1
    # A2 is not used in the lattice convention
    A2 = R2/RV * hA1
    # conversion from A_1, A_2 to A_12
    ff['A12'] = ((ff['A1']*(mB + mV)**2* (mB**2 - mV**2 - q2)
              - A2*(mB**4 + (mV**2 - q2)**2 - 2*mB**2*(mV**2 + q2)))
              / (16.*mB*mV**2*(mB + mV)))
    ff['V'] = R1/RV * hA1
    ff = isgur_wise(process=process, q2=q2, ff=ff, par=par, scale=scale)
    return ff
