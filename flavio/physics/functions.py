import scipy
import mpmath

def zeta(x):
    """Riemann Zeta function"""
    return scipy.special.zeta(x, 1)

def li2(x):
    r"""Complex Dilogarithm.

    $Li_2(x)=Polylog(2,x)$
    """
    # for real x<=1: use the special scipy function which is 20x faster
    if isinstance(x,float) and x <= 1:
        return scipy.special.spence(1-x)
    return mpmath.fp.polylog(2,x)

def ei(x):
    return scipy.special.expi(x)
