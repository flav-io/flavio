"""Generic formulae for fermion decay rates."""

from flavio.physics.common import lambda_K
from math import sqrt, pi

def GammaFvf(M, mv, mf, gL, gR, gTL=0, gtTL=0, gTR=0, gtTR=0):
    """Generic decay width of a fermion to a vector boson and a fermion."""
    if mv == 0:
        if any([gL,gR]):
            amp2_V = ((abs(gL)**2 + abs(gR)**2) * (M**2 + mf**2)
                    - 8 * mf * M * ( gR*gL.conjugate() ).real)
        else:
            amp2_V = False
        if any([gTL,gtTL,gTR,gtTR]):
            amp2_T = ( (M**2 - mf**2)**2 * (
                        abs(gTL)**2 + abs(gTR)**2
                      + abs(gtTL)**2 + abs(gtTR)**2
                    - 2 *( (gTL*gtTL.conjugate()).real
                         - (gTR*gtTR.conjugate()).real ) ))
        else:
            amp2_T = False
        if amp2_V is not False and amp2_T is not False:
            amp2_I = ( 3*M * (M**2 - mf**2)
                *( gL*(gTR+gtTR).conjugate() + gR*(gTL-gtTL).conjugate() ).real
                + 3*mf * (mf**2 - M**2)
                *( gR*(gTR+gtTR).conjugate() + gL*(gTL-gtTL).conjugate() ).real )
        else:
            amp2_I = 0
    else:
        if any([gL,gR]):
            amp2_V = ((abs(gL)**2 + abs(gR)**2)
                    * ((mf**2 - M**2)**2 /mv**2 + mf**2 + M**2 - 2. *mv**2)
                    - 12 * mf * M * ( gR*gL.conjugate() ).real )/2.
        else:
            amp2_V = False
        if any([gTL,gtTL,gTR,gtTR]):
            amp2_T = mv**2/2*( (abs(gTL-gtTL)**2 + abs(gTR+gtTR)**2)
                    * (2*(mf**2 - M**2)**2 /mv**2 - mf**2 - M**2 - mv**2)
                    - 12 * mf * M * ( (gTR+gtTR)*(gTL-gtTL).conjugate() ).real )
        else:
           amp2_T = False
        if amp2_V is not False and amp2_T is not False:
            amp2_I = ( 3*M * (M**2 - mf**2 - mv**2)
                *( gL*(gTR+gtTR).conjugate() + gR*(gTL-gtTL).conjugate() ).real
                + 3*mf * (mf**2 - M**2 - mv**2)
                *( gR*(gTR+gtTR).conjugate() + gL*(gTL-gtTL).conjugate() ).real )
        else:
            amp2_I = 0
    amp2 = float(amp2_V) + float(amp2_T) + amp2_I
    return Gamma(amp2, M, mf, mv)

def GammaFsf(M, ms, mf, gL, gR):
    """Generic decay width of a fermion to a scalar and a fermion."""
    amp2 = ((abs(gL)**2 + abs(gR)**2) * (M**2 + mf**2 - ms**2)
            + 4 * M * mf *(gL.real * gR.real + gL.imag * gR.imag))/2.
    return Gamma(amp2, M, mf, ms)

def Gamma(amp2, M, m1, m2):
    return sqrt(lambda_K(M**2,m1**2,m2**2)) /(16 * pi * M**3) * amp2
