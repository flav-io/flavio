"""Functions for running of QCD and QED gauge couplings below the weak scale."""

import numpy as np
from math import pi


def beta_qcd(als, ale, mu, f):
    r"""Right-hand side of the QCD beta function written in the (unconventional) form
    $d \alpha_s /d\mu= \beta(\mu)$
    """
    #FIXME QED part only implemented for f=5
    b=np.zeros((3,3), dtype=float)
    b[0,0] = (33 - 2*f)/3
    b[1,0] =    (102 - (38*f)/3.)
    b[2,0] = (1428.5 - (5033*f)/18. + (325*f**2)/54.)
    b[0,1] = -((22)/(9))
    b[1,1] = -(308/27)
    b[0,2] = (4945/243)
    couplings = np.array([[
            (als/4./pi)**ps*(ale/4./pi)**pe
            for pe in range(3)]
        for ps in range(3)])
    return -1/2./pi/mu*als**2*(couplings*b).sum()

def beta_qed(ale, als, mu, f):
    r"""RHS of the QED beta function written in the (unconventional) form
    $d \alpha_e /d\mu= \beta(\mu)$
    """
    #FIXME only implemented for f=5
    b=np.zeros((2,2), dtype=float)
    b[0,0] = (80/9)
    b[1,0] = (464/27)
    b[0,1] = (176/9)
    couplings = np.array([[
            (ale/4./pi)**pe*(als/4./pi)**ps
        for ps in range(2)]
    for pe in range(2)])
    return 1/2./pi/mu*ale**2*(couplings*b).sum()

def beta_qcd_qed(alpha, mu, nf):
    r"""RHS of the QCD and QED beta function written in the (unconventional) form
    $$\frac{d}{d\mu} \vec\alpha = \vec\beta(\mu)$$
    where $\vec\alpha^T=(\alpha_s,\alpha_e)$
    """
    bs =  beta_qcd(alpha[0], alpha[1], mu, nf)
    be =  beta_qed(alpha[1], alpha[0], mu, nf)
    return np.array([bs, be])

def betafunctions_qcd_qed_nf(nf):
    return lambda x, mu: beta_qcd_qed(x, mu, nf)
