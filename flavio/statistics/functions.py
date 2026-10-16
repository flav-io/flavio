"""Auxiliary functions for statistics."""

import scipy.stats
import scipy.special
from functools import lru_cache
from math import sqrt

@lru_cache(maxsize=10)
def confidence_level(nsigma):
    r"""Return the confidence level corresponding to a number of sigmas,
    i.e. the probability contained in the normal distribution between $-n\sigma$
    and $+n\sigma$.

    Example: `confidence_level(1)` returns approximately 0.68."""
    return scipy.special.erf(nsigma/sqrt(2))

def delta_chi2(nsigma, dof):
    r"""Compute the $\Delta\chi^2$ for `dof` degrees of freedom corresponding
    to `nsigma` Gaussian standard deviations.

    Example: For `dof=2` and `nsigma=1`, the result is roughly 2.3."""
    if dof == 1:
        # that's trivial
        return nsigma**2
    # use survival functions (not the confidence level) to avoid
    # saturation at large nsigma
    return scipy.stats.chi2.isf(2*scipy.stats.norm.sf(nsigma), dof)

def pull(delta_chi2, dof):
    r"""Compute the pull in Gaussian standard deviations corresponding to
    a $\Delta\chi^2$ with `dof` degrees of freedom.

    Example: For `dof=2` and `delta_chi2=2.3`, the result is roughly 1.0.

    This function is the inverse of the function `delta_chi2()`"""
    if dof == 1:
        # that's trivial
        return sqrt(abs(delta_chi2))
    return scipy.stats.norm.isf(scipy.stats.chi2.sf(delta_chi2, dof)/2)

def pvalue(chi2, dof):
    r"""Compute the $p$-value corresponding to a $\chi^2$ with `dof` degrees
    of freedom."""
    return scipy.stats.chi2.sf(chi2, dof)
