import scipy.stats

def delta_chi2(nsigma, dof):
    r"""Compute the $\Delta\chi^2$ for `dof` degrees of freedom corresponding
    to `nsigma` Gaussian standard deviations.

    Example: For `dof=2` and `nsigma=1`, the result is roughly 2.3."""
    if dof == 1:
        return nsigma**2
    chi2_1dof = scipy.stats.chi2(1)
    chi2_ndof = scipy.stats.chi2(dof)
    cl_nsigma = chi2_1dof.cdf(nsigma**2) # this is roughly 0.68 for n_sigma=1 etc.
    return chi2_ndof.ppf(cl_nsigma)
