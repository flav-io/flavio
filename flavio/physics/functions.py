"""Special mathematical functions"""


import scipy
import mpmath

def zeta(x):
    """Riemann Zeta function"""
    return scipy.special.zeta(x, 1)

def li2(x):
    r"""Complex Dilogarithm"""
    # for real x<=1: use the special scipy function which is 20x faster
    if x.imag==0. and x.real <= 1:
        return scipy.special.spence(1-x.real)
    return mpmath.fp.polylog(2,x)

def ei(x):
    """Exponential integral function"""
    return scipy.special.expi(x)
