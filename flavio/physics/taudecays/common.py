"""Generic formulae for fermion decay rates."""

from flavio.physics.common import lambda_K
from math import sqrt, pi

def GammaFvf(M, mv, mf, gL, gR):
    """Generic decay width of a fermion to a vector boson and a fermion."""
    amp2 = ((abs(gL)**2 + abs(gR)**2)
            * ((mf**2 - M**2)**2 /mv**2 + mf**2 + M**2 - 2. *mv**2)
            - 12 * mf * M * (gL.real * gR.real + gL.imag * gR.imag))/2.
    return Gamma(amp2, M, mf, mv)

def GammaFsf(M, ms, mf, gL, gR):
    """Generic decay width of a fermion to a scalar and a fermion."""
    amp2 = ((abs(gL)**2 + abs(gR)**2) * (M**2 + mf**2 - ms**2)
            + 4 * M * mf *(gL.real * gR.real + gL.imag * gR.imag))/2.
    return Gamma(amp2, M, mf, ms)

def Gamma(amp2, M, m1, m2):
    return sqrt(lambda_K(M**2,m1**2,m2**2)) /(16 * pi * M**3) * amp2
