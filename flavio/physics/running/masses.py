"""Functions for running of quark masses.

This module is based on the formulas in the `RunDec` papers, arXiv:hep-ph/0004189 and arXiv:1201.6149"""

from math import log, pi
import numpy as np
from flavio.math.functions import zeta
from flavio.physics.running.masses import zeta


def gamma0_qcd(nf):
    return 1.0
def gamma1_qcd(nf):
    return (202./3.-20.*nf/9.)/16.
def gamma2_qcd(nf):
    return (1249. + (-2216./27. - 160.*zeta(3)/3.)*nf-140.*nf*nf/81.)/64.
def gamma3_qcd(nf):
    return ((4603055./162. + 135680.*zeta(3)/27. - 8800.*zeta(5) +
          (-91723./27. - 34192.*zeta(3)/9. +
          880.*zeta(4) + 18400.*zeta(5)/9.)*nf +
          (5242./243. + 800.*zeta(3)/9. - 160.*zeta(4)/3.)*nf**2 +
          (-332./243. + 64.*zeta(3)/27.)*nf**3)/256.)

def gamma_qcd(mq, als, mu, f):
    r"""RHS of the QCD gamma function written in the (unconventional) form
    $d/d\mu m = gamma(\mu)$
    """
    g0 = gamma0_qcd(f)*(als/pi)**1
    g1 = gamma1_qcd(f)*(als/pi)**2
    g2 = gamma2_qcd(f)*(als/pi)**3
    g3 = gamma3_qcd(f)*(als/pi)**4
    return -2*mq/mu*(g0 + g1 + g2 + g3)


# (19) of arXiv:1107.3100v1
def fKsFromMs1(Mu, M, Nf):
    return -(4/3.)* (1- (4/3) * (Mu/M) - (Mu**2/(2*M**2)) )
def fKsFromMs2(Mu, M, Nf):
    b0 = 11 - 2*Nf/3.
    return ((((1/3.)*log(M/(2*Mu))+13/18.)*b0 - pi**2/3. + 23/18.)*Mu**2/M**2
            + (((8/9.)*log(M/(2*Mu))+64/27.)*b0 - 8*pi**2/9. + 92/27.) * Mu/M
            - (pi**2/12. + 71/96.)*b0
            + zeta(3)/6. - pi**2/9. * log(2) + 7*pi**2/12. + 23/72.
            )
# from (A.8) of hep-ph/0302262v1
def fKsFromMs3(Mu, M, Nf):
    b0 = 11 - 2*Nf/3.
    return -(b0/2.)**2*(2353/2592.+13/36.*pi**2+7/6.*zeta(3)
            -16/9.*Mu/M*((log(M/(2*Mu))+8/3.)**2+67/36.-pi**2/6.)
            -2/3.*Mu**2/M**2*((log(M/(2*Mu))+13/6.)**2+10/9.-pi**2/6.))

def mKS2mMS(M, Nf, asM, Mu, nl):
    s = np.zeros(4)
    s[0] = 1.
    s[1] = (asM/pi) * fKsFromMs1(Mu, M, Nf)
    s[2] = (asM/pi)**2 * fKsFromMs2(Mu, M, Nf)
    s[3] = (asM/pi)**3 * fKsFromMs3(Mu, M, Nf)
    r = s[:nl+1].sum()
    return M * r

def mMS2mKS(MS, Nf, asM, Mu, nl):
    def convert(M):
        s = np.zeros(4)
        s[0] = 1.
        s[1] = -(asM/pi) * fKsFromMs1(Mu, M, Nf)
        s[2] = -(asM/pi)**2 * fKsFromMs2(Mu, M, Nf)
        # properly invert the relation to O(asM**2)
        s[2] = s[2] + s[1]**2
        s[3] = -(asM/pi)**3 * fKsFromMs3(Mu, M, Nf)
        r = s[:nl+1].sum()
        return MS * r
    # iterate twice
    Mtmp = convert(MS)
    Mtmp = convert(Mtmp)
    return convert (Mtmp)
