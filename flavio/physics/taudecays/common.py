"""Generic formulae for fermion decay rates."""

from flavio.physics.common import lambda_K
from math import sqrt, pi

def GammaFvf(M, mv, mf, gL, gR, gTL=0, gtTL=0, gTR=0, gtTR=0):
    """Generic decay width of a fermion to a vector boson and a fermion."""
    amp2_V = ((abs(gL)**2 + abs(gR)**2)
            * ((mf**2 - M**2)**2 /mv**2 + mf**2 + M**2 - 2. *mv**2)
            - 12 * mf * M * (gL.real * gR.real + gL.imag * gR.imag))/2.
    if any([gTL,gtTL,gTR,gtTR]):
        amp2_T = mv**2/2*( (abs(gTL-gtTL)**2 + abs(gTR+gtTR)**2)
                * (2*(mf**2 - M**2)**2 /mv**2 - mf**2 - M**2 - mv**2)
                - 12 * mf * M * ( (gTL-gtTL).real * (gTR+gtTR).real
                                + (gTL-gtTL).imag * (gTR+gtTR).imag ) )
        I = ( 3*M * (mv**2 + mf**2 - M**2)
                *( gL.real*(gTR+gtTR).real + gR.real*(gTL-gtTL).real
                 + gL.imag*(gTR+gtTR).imag + gR.imag*(gTL-gtTL).imag)
            + 3*mf * (mv**2 + M**2 - mf**2)
                *( gL.real*(gTL-gtTL).real + gR.real*(gTR+gtTR).real
                 + gL.imag*(gTL-gtTL).imag + gR.imag*(gTR+gtTR).imag) )
        amp2 = amp2_V + amp2_T + I
    else:
        amp2 = amp2_V
    return Gamma(amp2, M, mf, mv)

def GammaFsf(M, ms, mf, gL, gR):
    """Generic decay width of a fermion to a scalar and a fermion."""
    amp2 = ((abs(gL)**2 + abs(gR)**2) * (M**2 + mf**2 - ms**2)
            + 4 * M * mf *(gL.real * gR.real + gL.imag * gR.imag))/2.
    return Gamma(amp2, M, mf, ms)

def Gamma(amp2, M, m1, m2):
    return sqrt(lambda_K(M**2,m1**2,m2**2)) /(16 * pi * M**3) * amp2
