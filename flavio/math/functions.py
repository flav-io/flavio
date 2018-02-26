"""Special mathematical functions"""


import scipy
import math
import numpy as np

def zeta(x):
    """Riemann Zeta function"""
    return scipy.special.zeta(x, 1)

def li2(x):
    r"""Complex Dilogarithm"""
    return scipy.special.spence(1-x)

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
