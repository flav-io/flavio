"""Special mathematical functions"""


import scipy
import mpmath
import math
import numpy as np

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
    if isinstance(x, float):
        _x = x
    else:
        _x = np.asarray(x)
    return -(_x-mu)**2/sigma**2/2 - math.log(math.sqrt(2*math.pi)*sigma)

def normal_pdf(x, mu, sigma):
    """PDF of the normal distribution"""
    # this turns out to be 2 orders of magnitude faster than scipy.stats.norm.logpdf
    if isinstance(x, float):
        _x = x
    else:
        _x = np.asarray(x)
    return np.exp(-(_x-mu)**2/sigma**2/2)/(np.sqrt(2*math.pi)*sigma)
