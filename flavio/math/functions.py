"""Special mathematical functions"""


import scipy
import mpmath
import math

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

def normal_logpdf(x, mu, sigma):
    """Logarithm of the PDF of the normal distribution"""
    # this turns out to be 2 orders of magnitude faster than scipy.stats.norm.logpdf
    return -(x-m)**2/s**2/2 - math.log(math.sqrt(2*math.pi)*s)
