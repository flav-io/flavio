"""Functions for conversion of quark masses not contained in RunDec."""

from math import log, pi
import numpy as np
from flavio.math.functions import zeta
from flavio.physics.running.masses import zeta


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
