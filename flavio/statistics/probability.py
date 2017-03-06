import numpy as np
import scipy.stats
from scipy.interpolate import interp1d, RegularGridInterpolator
import scipy.signal
import math
from flavio.math.functions import normal_logpdf, normal_pdf
from flavio.statistics.functions import confidence_level
import warnings

class ProbabilityDistribution(object):
    """Common base class for all probability distributions"""

    def __init__(self, central_value, support):
        self.central_value = central_value
        self.support = support

    def get_central(self):
        return self.central_value

    @property
    def error_left(self):
        """Return the lower error"""
        return self.get_error_left()

    @property
    def error_right(self):
        """Return the upper error"""
        return self.get_error_right()


class UniformDistribution(ProbabilityDistribution):
    """Distribution with constant PDF in a range and zero otherwise."""

    def __init__(self, central_value, half_range):
        """Initialize the distribution.

        Parameters:

        - central_value: arithmetic mean of the upper and lower range boundaries
        - half_range: half the difference of upper and lower range boundaries

        Example:

        central_value = 5 and half_range = 3 leads to the range [2, 8].
        """
        self.half_range = half_range
        self.range = (central_value - half_range,
                      central_value + half_range)
        super().__init__(central_value, support=self.range)

    def get_random(self, size=None):
        return np.random.uniform(self.range[0], self.range[1], size)

    def _logpdf(self, x):
        if x < self.range[0] or x >= self.range[1]:
            return -np.inf
        else:
            return -math.log(2 * self.half_range)

    def logpdf(self, x):
        _lpvect = np.vectorize(self._logpdf)
        return _lpvect(x)

    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        return confidence_level(nsigma) * self.half_range

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        return confidence_level(nsigma) * self.half_range


class DeltaDistribution(ProbabilityDistribution):
    """Delta Distrubution that is non-vanishing only at a single point."""

    def __init__(self, central_value):
        """Initialize the distribution.

        Parameters:

        - central_value: point where the PDF does not vanish.
        """
        super().__init__(central_value, support=(central_value, central_value))

    def get_random(self, size=None):
        if size is None:
            return self.central_value
        else:
            return self.central_value * np.ones(size)

    def logpdf(self, x):
        if np.ndim(x) == 0:
            if x == self.central_value:
                return 0.
            else:
                return -np.inf
        y = -np.inf*np.ones(np.asarray(x).shape)
        y[np.asarray(x) == self.central_value] = 0
        return y

    def get_error_left(self, *args, **kwargs):
        return 0

    def get_error_right(self, *args, **kwargs):
        return 0


class NormalDistribution(ProbabilityDistribution):
    """Univariate normal or Gaussian distribution."""

    def __init__(self, central_value, standard_deviation):
        """Initialize the distribution.

        Parameters:

        - central_value: location (mode and mean)
        - standard_deviation: standard deviation
        """
        super().__init__(central_value,
                         support=(central_value - 6 * standard_deviation,
                                  central_value + 6 * standard_deviation))
        if standard_deviation <= 0:
            raise ValueError("Standard deviation must be positive number")
        self.standard_deviation = standard_deviation

    def get_random(self, size=None):
        return np.random.normal(self.central_value, self.standard_deviation, size)

    def logpdf(self, x):
        return normal_logpdf(x, self.central_value, self.standard_deviation)

    def pdf(self, x):
        return normal_pdf(x, self.central_value, self.standard_deviation)

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, self.central_value, self.standard_deviation)

    def ppf(self, x):
        return scipy.stats.norm.ppf(x, self.central_value, self.standard_deviation)

    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        return nsigma * self.standard_deviation

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        return nsigma * self.standard_deviation


class AsymmetricNormalDistribution(ProbabilityDistribution):
    """An asymmetric normal distribution obtained by gluing together two
    half-Gaussians and demanding the PDF to be continuous."""

    def __init__(self, central_value, right_deviation, left_deviation):
        """Initialize the distribution.

        Parameters:

        - central_value: mode of the distribution (not equal to its mean!)
        - right_deviation: standard deviation of the upper half-Gaussian
        - left_deviation: standard deviation of the lower half-Gaussian
        """
        super().__init__(central_value,
                         support=(central_value - 6 * left_deviation,
                                  central_value + 6 * right_deviation))
        if right_deviation <= 0 or left_deviation <= 0:
            raise ValueError(
                "Left and right standard deviations must be positive numbers")
        self.right_deviation = right_deviation
        self.left_deviation = left_deviation
        self.p_right = normal_pdf(
            self.central_value, self.central_value, self.right_deviation)
        self.p_left = normal_pdf(
            self.central_value, self.central_value, self.left_deviation)

    def get_random(self, size=None):
        if size is None:
            return self._get_random()
        else:
            return np.array([self._get_random() for i in range(size)])

    def _get_random(self):
        r = np.random.uniform()
        a = abs(self.left_deviation /
                (self.right_deviation + self.left_deviation))
        if r > a:
            x = abs(np.random.normal(0, self.right_deviation))
            return self.central_value + x
        else:
            x = abs(np.random.normal(0, self.left_deviation))
            return self.central_value - x

    def _logpdf(self, x):
        # values of the PDF at the central value
        if x < self.central_value:
            # left-hand side: scale factor
            r = 2 * self.p_right / (self.p_left + self.p_right)
            return math.log(r) + normal_logpdf(x, self.central_value, self.left_deviation)
        else:
            # left-hand side: scale factor
            r = 2 * self.p_left / (self.p_left + self.p_right)
            return math.log(r) + normal_logpdf(x, self.central_value, self.right_deviation)

    def logpdf(self, x):
        _lpvect = np.vectorize(self._logpdf)
        return _lpvect(x)

    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        return nsigma * self.left_deviation

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        return nsigma * self.right_deviation


class HalfNormalDistribution(ProbabilityDistribution):
    """Half-normal distribution with zero PDF above or below the mode."""

    def __init__(self, central_value, standard_deviation):
        """Initialize the distribution.

        Parameters:

        - central_value: mode of the distribution.
        - standard_deviation:
          If positive, the PDF is zero below central_value and (twice) that of
          a Gaussian with this standard deviation above.
          If negative, the PDF is zero above central_value and (twice) that of
          a Gaussian with standard deviation equal to abs(standard_deviation)
          below.
        """

        super().__init__(central_value,
                         support=sorted((central_value,
                                         central_value + 6 * standard_deviation)))
        if standard_deviation == 0:
            raise ValueError("Standard deviation must be non-zero number")
        self.standard_deviation = standard_deviation

    def get_random(self, size=None):
        return self.central_value + np.sign(self.standard_deviation) * abs(np.random.normal(0, abs(self.standard_deviation), size))

    def _logpdf(self, x):
        if np.sign(self.standard_deviation) * (x - self.central_value) < 0:
            return -np.inf
        else:
            return math.log(2) + normal_logpdf(x, self.central_value, abs(self.standard_deviation))

    def logpdf(self, x):
        _lpvect = np.vectorize(self._logpdf)
        return _lpvect(x)

    def cdf(self, x):
        norm = scipy.stats.norm(loc=self.central_value,
                                scale=self.standard_deviation)
        cdf_0 = norm.cdf(0)
        cdf_x = norm.cdf(x)
        cdf = (cdf_x - cdf_0)/(1 - cdf_0)
        return np.piecewise(
                    np.asarray(x, dtype=float),
                    [x<0, x>=0],
                    [0., cdf]) # return 0 for negative x


    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        if self.standard_deviation >= 0:
            return 0
        else:
            return nsigma * (-self.standard_deviation)  # return a positive value!

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        if self.standard_deviation <= 0:
            return 0
        else:
            return nsigma * self.standard_deviation


class GaussianUpperLimit(HalfNormalDistribution):
    """Upper limit defined as a half-normal distribution."""

    def __init__(self, limit, confidence_level):
        """Initialize the distribution.

        Parameters:

        - limit: value of the upper limit
        - confidence_level: confidence_level of the upper limit. Float between
          0 and 1.
        """
        if confidence_level > 1 or confidence_level < 0:
            raise ValueError("Confidence level should be between 0 und 1")
        if limit <= 0:
            raise ValueError("The upper limit should be a positive number")
        super().__init__(central_value=0,
                         standard_deviation=self.get_standard_deviation(limit, confidence_level))
        self.limit = limit
        self.confidence_level = confidence_level

    def get_standard_deviation(self, limit, confidence_level):
        """Convert the confidence level into a Gaussian standard deviation"""
        return limit / scipy.stats.norm.ppf(0.5 + confidence_level / 2.)

class GammaDistribution(ProbabilityDistribution):
    r"""A Gamma distribution defined like the `gamma` distribution in
    `scipy.stats` (with parameters `a`, `loc`, `scale`).

    The `central_value` attribute returns the location of the mode.
    """

    def __init__(self, a, loc, scale):
        if loc > 0:
            raise ValueError("loc must be negative or zero")
        # "frozen" scipy distribution object
        self.scipy_dist = scipy.stats.gamma(a=a, loc=loc, scale=scale)
        mode = loc + (a-1)*scale
        # support extends until the CDF is roughly "6 sigma"
        support_limit = self.scipy_dist.ppf(1-2e-9)
        super().__init__(central_value=mode, # the mode
                         support=(loc, support_limit))
        self.a = a
        self.loc = loc
        self.scale = scale

    def get_random(self, size):
        return self.scipy_dist.rvs(size=size)

    def cdf(self, x):
        return self.scipy_dist.cdf(x)

    def ppf(self, x):
        return self.scipy_dist.ppf(x)

    def logpdf(self, x):
        return self.scipy_dist.logpdf(x)

    def _find_error_cdf(self, confidence_level):
        # find the value of the CDF at the position of the left boundary
        # of the `confidence_level`% CL range by demanding that the value
        # of the PDF is the same at the two boundaries
        def x_left(a):
            return self.ppf(a)
        def x_right(a):
            return self.ppf(a + confidence_level)
        def diff_logpdf(a):
            logpdf_x_left = self.logpdf(x_left(a))
            logpdf_x_right = self.logpdf(x_right(a))
            return logpdf_x_left - logpdf_x_right
        return scipy.optimize.brentq(diff_logpdf, 0,  1 - confidence_level-1e-6)

    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        a = self._find_error_cdf(confidence_level(nsigma))
        return self.central_value - self.ppf(a)

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        a = self._find_error_cdf(confidence_level(nsigma))
        return self.ppf(a + confidence_level(nsigma)) - self.central_value

class GammaDistributionPositive(ProbabilityDistribution):
    r"""A Gamma distribution defined like the `gamma` distribution in
    `scipy.stats` (with parameters `a`, `loc`, `scale`), but restricted to
    positive values for x and correspondingly rescaled PDF.

    The `central_value` attribute returns the location of the mode.
    """

    def __init__(self, a, loc, scale):
        if loc > 0:
            raise ValueError("loc must be negative or zero")
        # "frozen" scipy distribution object (without restricting x>0!)
        self.scipy_dist = scipy.stats.gamma(a=a, loc=loc, scale=scale)
        mode = loc + (a-1)*scale
        if mode < 0:
            mode = 0
        # support extends until the CDF is roughly "6 sigma", assuming x>0
        support_limit = self.scipy_dist.ppf(1-2e-9*(1-self.scipy_dist.cdf(0)))
        super().__init__(central_value=mode, # the mode
                         support=(0, support_limit))
        self.a = a
        self.loc = loc
        self.scale = scale
        # scale factor for PDF to account for x>0
        self._pdf_scale = 1/(1 - self.scipy_dist.cdf(0))

    def get_random(self, size=None):
        if size is None:
            return self._get_random()
        else:
            # some iteration necessary as discarding negative values
            # might lead to too small size
            r = np.array([], dtype=float)
            while len(r) < size:
                r = np.concatenate((r, self._get_random(size=2*size)))
            return r[:size]

    def _get_random(self, size):
        r = self.scipy_dist.rvs(size=size)
        return r[(r >= 0)]

    def cdf(self, x):
        cdf0 = self.scipy_dist.cdf(0)
        cdf = (self.scipy_dist.cdf(x) - cdf0)/(1-cdf0)
        return np.piecewise(
                    np.asarray(x, dtype=float),
                    [x<0, x>=0],
                    [0., cdf]) # return 0 for negative x

    def ppf(self, x):
        cdf0 = self.scipy_dist.cdf(0)
        return self.scipy_dist.ppf((1-cdf0)*x +  cdf0)

    def logpdf(self, x):
        # return -inf for negative x values
        inf0 = np.piecewise(np.asarray(x, dtype=float), [x<0, x>=0], [-np.inf, 0.])
        return inf0 + self.scipy_dist.logpdf(x) + np.log(self._pdf_scale)

    def _find_error_cdf(self, confidence_level):
        # find the value of the CDF at the position of the left boundary
        # of the `confidence_level`% CL range by demanding that the value
        # of the PDF is the same at the two boundaries
        def x_left(a):
            return self.ppf(a)
        def x_right(a):
            return self.ppf(a + confidence_level)
        def diff_logpdf(a):
            logpdf_x_left = self.logpdf(x_left(a))
            logpdf_x_right = self.logpdf(x_right(a))
            return logpdf_x_left - logpdf_x_right
        return scipy.optimize.brentq(diff_logpdf, 0,  1 - confidence_level-1e-6)

    def get_error_left(self, nsigma=1):
        """Return the lower error"""
        if self.logpdf(0) > self.logpdf(self.ppf(confidence_level(nsigma))):
            # look at a one-sided 1 sigma range. If the PDF at 0
            # is smaller than the PDF at the boundary of this range, it means
            # that the left-hand error is not meaningful to define.
            return self.central_value
        else:
            a = self._find_error_cdf(confidence_level(nsigma))
            return self.central_value - self.ppf(a)

    def get_error_right(self, nsigma=1):
        """Return the upper error"""
        one_sided_error = self.ppf(confidence_level(nsigma))
        if self.logpdf(0) > self.logpdf(one_sided_error):
            # look at a one-sided 1 sigma range. If the PDF at 0
            # is smaller than the PDF at the boundary of this range, return the
            # boundary of the range as the right-hand error
            return one_sided_error
        else:
            a = self._find_error_cdf(confidence_level(nsigma))
            return self.ppf(a + confidence_level(nsigma)) - self.central_value

class GammaUpperLimit(GammaDistributionPositive):
    r"""Gamma distribution with x restricted to be positive appropriate for
    a positive quantitity obtained from a low-statistics counting experiment,
    e.g. a rare decay rate, given an upper limit on x."""

    def __init__(self, counts_total, counts_background, limit, confidence_level):
        r"""Initialize the distribution.

        Parameters:

        - counts_total: observed total number (signal and background) of counts.
        - counts_background: number of expected background counts, assumed to be
          known.
        - limit: upper limit on x, which is proportional (with a positive
          proportionality factor) to the number of signal events.
        - confidence_level: confidence level of the upper limit, i.e. the value
          of the CDF at the limit. Float between 0 and 1. Frequently used values
          are 0.90 and 0.95.
        """
        if confidence_level > 1 or confidence_level < 0:
            raise ValueError("Confidence level should be between 0 und 1")
        if limit <= 0:
            raise ValueError("The upper limit should be a positive number")
        if counts_total <= 0:
            raise ValueError("counts_total should be a positive number")
        if counts_background <= 0:
            raise ValueError("counts_background should be a positive number")
        self.limit = limit
        self.confidence_level = confidence_level
        self.counts_total = counts_total
        self.counts_background = counts_background
        a, loc, scale = self._get_a_loc_scale()
        super().__init__(a=a, loc=loc, scale=scale)

    def _get_a_loc_scale(self):
        """Convert the counts and limit to the input parameters needed for
        GammaDistributionPositive"""
        a = self.counts_total + 1
        loc_unscaled = -self.counts_background
        dist_unscaled = GammaDistributionPositive(a=a, loc=loc_unscaled, scale=1)
        limit_unscaled = dist_unscaled.ppf(self.confidence_level)
        # rescale
        scale = self.limit/limit_unscaled
        loc = -self.counts_background*scale
        return a, loc, scale

class NumericalDistribution(ProbabilityDistribution):
    """Univariate distribution defined in terms of numerical values for the
    PDF."""

    def __init__(self, x, y, central_value=None):
        """Initialize a 1D numerical distribution.

        Parameters:

        - `x`: x-axis values. Must be a 1D array of real values in strictly
          ascending order (but not necessarily evenly spaced)
        - `y`: PDF values. Must be a 1D array of real positive values with the
          same length as `x`
        - central_value: if None (default), will be set to the mode of the
          distribution, i.e. the x-value where y is largest (by looking up
          the input arrays, i.e. without interpolation!)
        """
        self.x = x
        self.y = y
        if central_value is not None:
            if x[0] <= central_value <= x[-1]:
                super().__init__(central_value=central_value,
                                 support=(x[0], x[-1]))
            else:
                raise ValueError("Central value must be within range provided")
        else:
            mode = x[np.argmax(y)]
            super().__init__(central_value=mode, support=(x[0], x[-1]))
        self.y_norm = y /  np.trapz(y, x=x)  # normalize PDF to 1
        self.y_norm[self.y_norm < 0] = 0
        self.pdf_interp = interp1d(x, self.y_norm,
                                        fill_value=0, bounds_error=False)
        _cdf = np.zeros(len(x))
        _cdf[1:] = np.cumsum(self.y_norm[:-1] * np.diff(x))
        _cdf = _cdf/_cdf[-1] # normalize CDF to 1
        self.ppf_interp = interp1d(_cdf, x)
        self.cdf_interp = interp1d(x, _cdf)

    def get_random(self, size=None):
        """Draw a random number from the distribution.

        If size is not None but an integer N, return an array of N numbers."""
        r = np.random.uniform(size=size)
        return self.ppf_interp(r)

    def ppf(self, x):
        return self.ppf_interp(x)

    def cdf(self, x):
        return self.cdf_interp(x)

    def pdf(self, x):
        return self.pdf_interp(x)

    def logpdf(self, x):
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log(self.pdf_interp(x))

    def _find_error_cdf(self, confidence_level):
        # find the value of the CDF at the position of the left boundary
        # of the `confidence_level`% CL range by demanding that the value
        # of the PDF is the same at the two boundaries
        def x_left(a):
            return self.ppf(a)
        def x_right(a):
            return self.ppf(a + confidence_level)
        def diff_logpdf(a):
            logpdf_x_left = self.logpdf(x_left(a))
            logpdf_x_right = self.logpdf(x_right(a))
            return logpdf_x_left - logpdf_x_right
        return scipy.optimize.brentq(diff_logpdf, 0,  1 - confidence_level-1e-6)

    def get_error_left(self, nsigma=1, method='central'):
        """Return the lower error.

        'method' should be one of:

        - 'central' for a central interval (same probability on both sides of
          the central value)
        - 'hpd' for highest posterior density, i.e. probability is larger inside
          the interval than outside
        - 'limit' for a one-sided error, i.e. a lower limit"""
        if method == 'limit':
            return self.central_value - self.ppf(1 - confidence_level(nsigma))
        cdf_central = self.cdf(self.central_value)
        err_left = self.central_value - self.ppf(cdf_central * (1 - confidence_level(nsigma)))
        if method == 'central':
            return err_left
        elif method == 'hpd':
            if self.pdf(self.central_value + self.get_error_right(method='central')) == self.pdf(self.central_value - err_left):
                return err_left
            try:
                a = self._find_error_cdf(confidence_level(nsigma))
            except ValueError:
                return np.nan
            return self.central_value - self.ppf(a)
        else:
            raise ValueError("Method " + str(method) + " unknown")

    def get_error_right(self, nsigma=1, method='central'):
        """Return the upper error

        'method' should be one of:

        - 'central' for a central interval (same probability on both sides of
          the central value)
        - 'hpd' for highest posterior density, i.e. probability is larger inside
          the interval than outside
        - 'limit' for a one-sided error, i.e. an upper limit"""
        if method == 'limit':
            return self.ppf(confidence_level(nsigma)) - self.central_value
        cdf_central = self.cdf(self.central_value)
        err_right = self.ppf(cdf_central + (1 - cdf_central) * confidence_level(nsigma)) - self.central_value
        if method == 'central':
            return err_right
        elif method == 'hpd':
            if self.pdf(self.central_value - self.get_error_left(method='central')) == self.pdf(self.central_value + err_right):
                return err_right
            try:
                a = self._find_error_cdf(confidence_level(nsigma))
            except ValueError:
                return np.nan
            return self.ppf(a + confidence_level(nsigma)) - self.central_value
        else:
            raise ValueError("Method " + str(method) + " unknown")

    @classmethod
    def from_pd(cls, pd, nsteps=1000):
        if isinstance(pd, NumericalDistribution):
            return pd
        _x = np.linspace(pd.support[0], pd.support[-1], nsteps)
        _y = np.exp(pd.logpdf(_x))
        return cls(central_value=pd.central_value, x=_x, y=_y)

class GeneralGammaUpperLimit(NumericalDistribution):
    r"""Distribution appropriate for
    a positive quantitity obtained from a low-statistics counting experiment,
    e.g. a rare decay rate, given an upper limit on x.
    The difference to `GammaUpperLimit` is that this class also allows to
    specify an uncertainty on the number of background events. The result
    is a numerical distribution obtained from the convolution of a normal
    distribution (for the background uncertainty) and a gamma distribution,
    restricted to positive values.
    """

    def __init__(self,
                 limit, confidence_level,
                 counts_total=None,
                 counts_background=None,
                 counts_signal=None,
                 background_variance=0):
        r"""Initialize the distribution.

        Parameters:

        Parameters:

        - `limit`: upper limit on x, which is proportional (with a positive
          proportionality factor) to the number of signal events.
        - `confidence_level`: confidence level of the upper limit, i.e. the value
          of the CDF at the limit. Float between 0 and 1. Frequently used values
          are 0.90 and 0.95.
        - `counts_total`: observed total number (signal and background) of counts.
        - `counts_background`: expected mean number of expected background counts
        - `counts_signal`: mean obseved number of signal events
        - `background_variance`: standard deviation of the expected number of
          background events

        Of the three parameters `counts_total`, `counts_background`, and
        `counts_signal`, exactly two should be specified. The third one will
        be determined from the relation

        `counts_total = counts_signal + counts_background`

        Note that if `background_variance=0`, it makes more sense to use
        `GammaUpperLimit`, which is equivalent but analytical rather than
        numerical.
        """
        if confidence_level > 1 or confidence_level < 0:
            raise ValueError("Confidence level should be between 0 und 1")
        if limit <= 0:
            raise ValueError("The upper limit should be a positive number")
        if counts_total is not None and counts_total <= 0:
            raise ValueError("counts_total should be a positive number or None")
        if counts_background is not None and counts_background <= 0:
            raise ValueError("counts_background should be a positive number or None")
        if background_variance < 0:
            raise ValueError("background_variance should be a positive number")
        self.limit = limit
        self.confidence_level = confidence_level
        if [counts_total, counts_signal, counts_background].count(None) != 1:
            raise ValueError("You must specify exactly two of counts_total, counts_signal, counts_background")
        if counts_background is None:
            self.counts_background = counts_total - counts_signal
        else:
            self.counts_background = counts_background
        if counts_signal is None:
            self.counts_signal = counts_total - counts_background
        else:
            self.counts_signal = counts_signal
        if counts_total is None:
            self.counts_total = counts_signal + counts_background
        else:
            self.counts_total = counts_total
        self.background_variance = background_variance
        x, y = self._get_xy()
        if self.background_variance/self.counts_total <= 1/100.:
            warnings.warn("For vanishing or very small background variance, "
                          "it is safer to use GammaUpperLimit instead of "
                          "GeneralGammaUpperLimit to avoid numerical "
                          "instability.")
        super().__init__(x=x, y=y)

    def _get_xy(self):
        if self.background_variance == 0:
            # this is a bit pointless as in this case it makes more
            # sense to use GammaUpperLimit itself
            gamma_unscaled = GammaDistributionPositive(a = self.counts_total + 1,
                                                       loc = -self.counts_background,
                                                       scale = 1)
            num_unscaled = NumericalDistribution.from_pd(gamma_unscaled)
        else:
            # define a gamma distribution (with x>loc, not x>0!) and convolve
            # it with a Gaussian
            gamma_unscaled = GammaDistribution(a = self.counts_total + 1,
                                               loc = -self.counts_background,
                                               scale = 1)
            norm_bg = NormalDistribution(0, self.background_variance)
            num_unscaled = convolve_distributions([gamma_unscaled, norm_bg], central_values='sum')
        # now that we have convolved, cut off anything below x=0
        x = num_unscaled.x
        y = num_unscaled.y_norm
        y[x<0] = 0
        num_unscaled = NumericalDistribution(x, y)
        limit_unscaled = num_unscaled.ppf(self.confidence_level)
        # use the value of the limit to determine the scale factor
        scale_factor = self.limit/limit_unscaled
        x = x * scale_factor
        return x, y


class KernelDensityEstimate(NumericalDistribution):
    """Univariate kernel density estimate.

    Parameters:

    - `data`: 1D array
    - `kernel`: instance of `ProbabilityDistribution` used as smoothing kernel
    - `n_bins` (optional): number of bins used in the intermediate step. This normally
      does not have to be changed.
    """

    def __init__(self, data, kernel, n_bins=None):
        self.data = data
        assert kernel.central_value == 0, "Kernel density must have zero central value"
        self.kernel = kernel
        self.n = len(data)
        if n_bins is None:
            self.n_bins = min(1000, self.n)
        else:
            self.n_bins = n_bins
        y, x_edges = np.histogram(data, bins=self.n_bins, normed=True)
        x = (x_edges[:-1] + x_edges[1:])/2.
        self.y_raw = y
        self.raw_dist = NumericalDistribution(x, y)
        cdist = convolve_distributions([self.raw_dist, self.kernel], 'sum')
        super().__init__(cdist.x, cdist.y)


class GaussianKDE(KernelDensityEstimate):
    """Univariate Gaussian kernel density estimate.

    Parameters:

    - `data`: 1D array
    - `bandwidth` (optional): standard deviation of the Gaussian smoothing kernel.
       If not provided, Scott's rule is used to estimate it.
    - `n_bins` (optional): number of bins used in the intermediate step. This normally
      does not have to be changed.
    """

    def __init__(self, data, bandwidth=None, n_bins=None):
        if bandwidth is None:
            self.bandwidth = len(data)**(-1/5.) * np.std(data)
        else:
            self.bandwidth = bandwidth
        super().__init__(data=data,
                         kernel = NormalDistribution(0, self.bandwidth),
                         n_bins=n_bins)


class MultivariateNormalDistribution(ProbabilityDistribution):
    """A multivariate normal distribution.

    Methods:

    - get_random(size=None): get `size` random numbers (default: a single one)
    - logpdf(x, exclude=None): get the logarithm of the probability density
      function. If an iterable of integers is given for `exclude`, the parameters
      at these positions will be removed from the covariance before evaluating
      the PDF, effectively ignoring certain dimensions.

    Properties:

    - error_left, error_right: both return the vector of standard deviations
    """

    def __init__(self, central_value, covariance):
        """Initialize PDF instance.

        Parameters:

        - central_value: vector of means, shape (n)
        - covariance: covariance matrix, shape (n,n)
        """
        super().__init__(central_value, support=np.array([
                    np.asarray(central_value) - 6*np.sqrt(np.diag(covariance)),
                    np.asarray(central_value) + 6*np.sqrt(np.diag(covariance))
                    ]))
        self.covariance = covariance
        # to avoid ill-conditioned covariance matrices, all data are rescaled
        # by the inverse variances
        self.err = np.sqrt(np.diag(self.covariance))
        self.scaled_covariance = self.covariance / np.outer(self.err, self.err)
        assert np.all(np.linalg.eigvals(self.scaled_covariance) >
                      0), "The covariance matrix is not positive definite!" + str(covariance)

    def get_random(self, size=None):
        """Get `size` random numbers (default: a single one)"""
        return np.random.multivariate_normal(self.central_value, self.covariance, size)

    def logpdf(self, x, exclude=None):
        """Get the logarithm of the probability density function.

        Parameters:

        - x: vector; position at which PDF should be evaluated
        - exclude: optional; if an iterable of integers is given, the parameters
          at these positions will be removed from the covariance before
          evaluating the PDF, effectively ignoring certain dimensions.
        """
        if exclude is not None:
            # if parameters are to be excluded, construct a temporary
            # distribution with reduced mean vector and covariance matrix
            # and call its logpdf method
            _cent_ex = np.delete(self.central_value, exclude)
            _cov_ex = np.delete(
                np.delete(self.covariance, exclude, axis=0), exclude, axis=1)
            if len(_cent_ex) == 1:
                # if only 1 dimension remains, can use a univariate Gaussian
                _dist_ex = NormalDistribution(
                    central_value=_cent_ex[0], standard_deviation=np.sqrt(_cov_ex[0, 0]))
                return _dist_ex.logpdf(x)
            else:
                # if more than 1 dimension remains, use a (smaller)
                # multivariate Gaussian
                _dist_ex = MultivariateNormalDistribution(
                    central_value=_cent_ex, covariance=_cov_ex)
                return _dist_ex.logpdf(x, exclude=None)
        # undoing the rescaling of the covariance
        pdf_scaled = scipy.stats.multivariate_normal.logpdf(
            x / self.err, self.central_value / self.err, self.scaled_covariance)
        sign, logdet = np.linalg.slogdet(self.covariance)
        return pdf_scaled + (np.linalg.slogdet(self.scaled_covariance)[1] - np.linalg.slogdet(self.covariance)[1]) / 2.

    def get_error_left(self, nsigma=1):
        """Return the lower errors"""
        return nsigma * self.err

    def get_error_right(self, nsigma=1):
        """Return the upper errors"""
        return nsigma * self.err


class MultivariateNumericalDistribution(ProbabilityDistribution):
    """A multivariate distribution with PDF specified numerically."""

    def __init__(self, xi, y, central_value=None):
        """Initialize a multivariate numerical distribution.

        Parameters:

        - `xi`: for an N-dimensional distribution, a list of N 1D arrays
          specifiying the grid in N dimensions. The 1D arrays must contain
          real, evenly spaced values in strictly ascending order (but the
          spacing can be different for different dimensions). Any of the 1D
          arrays can also be given alternatively as a list of two numbers, which
          will be assumed to be the upper and lower boundaries, while the
          spacing will be determined from the shape of `y`.
        - `y`: PDF values on the grid defined by the `xi`. If the N `xi` have
          length M1, ..., MN, `y` has dimension (M1, ..., MN). This is the same
          shape as the grid obtained from `numpy.meshgrid(*xi, indexing='ij')`.
        - central_value: if None (default), will be set to the mode of the
          distribution, i.e. the N-dimensional xi-vector where y is largest
          (by looking up the input arrays, i.e. without interpolation!)
        """
        for x in xi:
            # check that grid spacings are even up to per mille precision
            d = np.diff(x)
            if abs(np.min(d)/np.max(d)-1) > 1e-3:
                raise ValueError("Grid must be evenly spaced per dimension")
        self.xi = list(xi)
        self.y = np.asarray(y)
        for i, x in enumerate(xi):
            if len(x) == 2:
                self.xi[i] = np.linspace(x[0], x[1], self.y.shape[i])
        if central_value is not None:
            super().__init__(central_value=central_value,
                             support=(np.asarray(self.xi).T[0], np.asarray(self.xi).T[-1]))
        else:
            # if no central value is specified, set it to the mode
            mode_index = (slice(None),) + np.unravel_index(self.y.argmax(), self.y.shape)
            mode = np.asarray(np.meshgrid(*self.xi, indexing='ij'))[mode_index]
            super().__init__(central_value=mode, support=None)
        _bin_volume = np.prod([x[1] - x[0] for x in self.xi])
        self.y_norm = self.y / np.sum(self.y) / _bin_volume  # normalize PDF to 1
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            # logy = np.nan_to_num(np.log(self.y_norm))
            logy = np.log(self.y_norm)
            logy[np.isneginf(logy)] = -1e100
            self.logpdf_interp = RegularGridInterpolator(self.xi, logy,
                                        fill_value=-np.inf, bounds_error=False)
        # the following is needed for get_random: initialize to None
        self._y_flat = None
        self._cdf_flat = None


    def get_random(self, size=None):
        """Draw a random number from the distribution.

        If size is not None but an integer N, return an array of N numbers.

        For the MultivariateNumericalDistribution, the PDF from which the
        random numbers are drawn is approximated to be piecewise constant in
        hypercubes around the points of the lattice spanned by the `xi`. A finer
        lattice spacing will lead to a smoother distribution of random numbers
        (but will also be slower).
        """

        if size is None:
            return self._get_random()
        else:
            return np.array([self._get_random() for i in range(size)])

    def _get_random(self):
        # if these have not been initialized, do it (once)
        if self._y_flat is None:
            # get a flattened array of the PDF
            self._y_flat = self.y.flatten()
        if self._cdf_flat is None:
            # get the (discrete) 1D CDF
            _cdf_flat =  np.cumsum(self._y_flat)
            # normalize to 1
            self._cdf_flat = _cdf_flat/_cdf_flat[-1]
        # draw a number between 0 and 1
        r = np.random.uniform()
        # find the index of the CDF-value closest to r
        i_r = np.argmin(np.abs(self._cdf_flat-r))
        indices = np.where(self.y == self._y_flat[i_r])
        i_bla = np.random.choice(len(indices[0]))
        index = tuple([a[i_bla] for a in indices])
        xi_r = [ self.xi[i][index[i]] for i in range(len(self.xi)) ]
        xi_diff = np.array([ X[1]-X[0] for X in self.xi ])
        return xi_r + np.random.uniform(low=-0.5, high=0.5, size=len(self.xi)) * xi_diff

    def logpdf(self, x, exclude=None):
        """Get the logarithm of the probability density function.

        Parameters:

        - x: vector; position at which PDF should be evaluated
        - exclude: optional; if an iterable of integers is given, the parameters
          at these positions will be ignored by maximizing the likelihood
          along the remaining directions, i.e., they will be "profiled out".
        """
        if exclude is not None:
            try:
                exclude = tuple(exclude)
            except TypeError:
                exclude = (exclude,)
            xi = np.delete(self.xi, tuple(exclude), axis=0)
            y = np.amax(self.y_norm, axis=tuple(exclude))
            cv = np.delete(self.central_value, tuple(exclude))
            if len(xi) == 1:
                # if there is just 1 dimension left, use univariate
                dist = NumericalDistribution(xi[0], y, cv)
            else:
                dist = MultivariateNumericalDistribution(xi, y, cv)
            return dist.logpdf(x)
        if np.asarray(x).shape == (len(self.central_value),):
            # return a scalar
            return self.logpdf_interp(x)[0]
        else:
            return self.logpdf_interp(x)

    def get_error_left(self, *args, **kwargs):
        raise NotImplementedError(
            "1D errors not implemented for multivariate numerical distributions")

    def get_error_right(self, *args, **kwargs):
        raise NotImplementedError(
            "1D errors not implemented for multivariate numerical distributions")

    @classmethod
    def from_pd(cls, pd, nsteps=100):
        _xi = np.array([np.linspace(pd.support[0, i], pd.support[-1, i], nsteps)
                        for i in range(len(pd.central_value))])
        ndim = len(_xi)
        _xlist = np.array(np.meshgrid(*_xi, indexing='ij')).reshape(ndim, nsteps**ndim).T
        _ylist = np.exp(pd.logpdf(_xlist))
        _y = _ylist.reshape(tuple(nsteps for i in range(ndim)))
        return cls(central_value=pd.central_value, xi=_xi, y=_y)


# Auxiliary functions

def convolve_distributions(probability_distributions, central_values='same'):
    """Combine a set of probability distributions by convoluting the PDFs.

    This function can be used in two different ways:

    - for `central_values='same'`, it can be used to combine uncertainties on a
    single parameter/observable expressed in terms of probability distributions
    with the same central value.
    - for `central_values='sum'`, it can be used to determine the probability
    distribution of a sum of random variables.

    The only difference between the two cases is a shift: for 'same', the
    central value of the convolution is the same as the original central value,
    for 'sum', it is the sum of the individual central values.

    `probability_distributions` must be a list of instances of descendants of
    `ProbabilityDistribution`.
    """
    if central_values not in ['same', 'sum']:
        raise ValueError("central_values must be either 'same' or 'sum'")
    def dim(x):
        # 1 for floats and length for arrays
        try:
            float(x)
        except:
            return len(x)
        else:
            return 1
    dims = [dim(p.central_value) for p in probability_distributions]
    assert all([d == dims[0] for d in dims]), "All distributions must have the same number of dimensions"
    if dims[0] == 1:
        return _convolve_distributions_univariate(probability_distributions, central_values)
    else:
        return _convolve_distributions_multivariate(probability_distributions, central_values)

def _convolve_distributions_univariate(probability_distributions, central_values='same'):
    """Combine a set of univariate probability distributions."""
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    if central_values == 'same':
        central_value = probability_distributions[0].central_value
        assert all(p.central_value == central_value for p in probability_distributions), \
            "Distributions must all have the same central value"

    # all delta dists
    deltas = [p for p in probability_distributions if isinstance(
        p, DeltaDistribution)]
    if central_values == 'sum' and deltas:
        raise NotImplementedError("Convolution of DeltaDistributions only implemented for equal central values")
    # central_values is 'same', we can instead just ignore the delta distributions!

    # all normal dists
    gaussians = [p for p in probability_distributions if isinstance(
        p, NormalDistribution)]

    # all other univariate dists
    others = list(set(probability_distributions) - set(gaussians) - set(deltas))

    if not others and not gaussians:
        # if there is only a delta (or more than one), just return it
        if central_values == 'same':
            return deltas[0]
        elif central_values == 'same':
            return DeltaDistribution(sum([p.central_value for p in deltas]))

    # let's combine the normal distributions into 1
    if gaussians:
        gaussian = _convolve_gaussians(gaussians, central_values=central_values)

    if gaussians and not others:
        # if there are only the gaussians, we are done.
        return gaussian
    else:
        # otherwise, we need to combine the (combined) gaussian with the others
        if gaussians:
            to_be_combined = others + [gaussian]
        else:
            to_be_combined = others
        # turn all distributions into numerical distributions!
        numerical = [NumericalDistribution.from_pd(p) for p in to_be_combined]
        return _convolve_numerical(numerical, central_values=central_values)


def _convolve_distributions_multivariate(probability_distributions, central_values='same'):
    """Combine a set of multivariate probability distributions."""
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    if central_values == 'same':
        central_value = probability_distributions[0].central_value
        assert all(p.central_value[i] == central_value[i] for p in probability_distributions for i in range(len(central_value))), \
            "Distributions must all have the same central value"

    for p in probability_distributions:
        if not ( isinstance(p, MultivariateNormalDistribution)
                 or isinstance(p, MultivariateNumericalDistribution) ):
            raise ValueError("Multivariate convolution only implemented "
                             "for normal and numerical distributions")

    # all normal dists
    gaussians = [p for p in probability_distributions if isinstance(
        p, MultivariateNormalDistribution)]

    # all numerical dists
    others = [p for p in probability_distributions if isinstance(
        p, MultivariateNumericalDistribution)]

    # let's combine the normal distributions into 1
    if gaussians:
        gaussian = _convolve_multivariate_gaussians(gaussians,
                                                  central_values=central_values)

    if gaussians and not others:
        # if there are only the gaussians, we are done.
        return gaussian
    else:
        # otherwise, we need to combine the (combined) gaussian with the others
        if len(others) > 1:
            NotImplementedError("Combining multivariate numerical distributions not implemented")
        else:
            num = _convolve_multivariate_gaussian_numerical(gaussian, others[0],
                                                  central_values=central_values)
            return num


def _convolve_gaussians(probability_distributions, central_values='same'):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, NormalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NormalDistribution"
    if central_values == 'same':
        central_value = probability_distributions[0].central_value  # central value of the first dist
        assert all(p.central_value == central_value for p in probability_distributions), \
            "Distrubtions must all have the same central value"
    elif central_values == 'sum':
        central_value = sum([p.central_value for p in probability_distributions])
    sigmas = np.array(
        [p.standard_deviation for p in probability_distributions])
    sigma = math.sqrt(np.sum(sigmas**2))
    return NormalDistribution(central_value=central_value, standard_deviation=sigma)


def _convolve_multivariate_gaussians(probability_distributions, central_values='same'):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, MultivariateNormalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of MultivariateNormalDistribution"
    if central_values == 'same':
        central_value = probability_distributions[0].central_value  # central value of the first dist
        assert all(p.central_value == central_value for p in probability_distributions), \
            "Distrubtions must all have the same central value"
    elif central_values == 'sum':
        central_value = np.sum([p.central_value for p in probability_distributions], axis=0)
    cov =  np.sum([p.covariance for p in probability_distributions], axis=0)
    return MultivariateNormalDistribution(central_value=central_value, covariance=cov)


def _convolve_numerical(probability_distributions, nsteps=1000, central_values='same'):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, NumericalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NumericalDistribution"
    if central_values == 'same':
        central_value = probability_distributions[0].central_value  # central value of the first dist
        assert all(p.central_value == central_value for p in probability_distributions), \
            "Distrubtions must all have the same central value"
    elif central_values == 'sum':
        central_value = sum([p.central_value for p in probability_distributions])
    # differences of individual central values from combined central value
    central_diffs = [central_value - p.central_value for p in probability_distributions]

    # (shifted appropriately)
    supports = (np.array([p.support for p in probability_distributions]).T + central_diffs).T
    support = (central_value - (central_value - supports[:, 0]).sum(),
               central_value - (central_value - supports[:, 1]).sum())
    delta = (support[1] - support[0]) / (nsteps - 1)
    x = np.linspace(support[0], support[1], nsteps)
    # position of the central value
    n_x_central = math.floor((central_value - support[0]) / delta)
    y = None
    for i, pd in enumerate(probability_distributions):
        y1 = np.exp(pd.logpdf(x - central_diffs[i])) * delta
        if y is None:
            # first step
            y = y1
        else:
            # convolution
            y = scipy.signal.fftconvolve(y, y1, 'full')
            # cut out the convolved signal at the right place
            y = y[n_x_central:nsteps + n_x_central]
    return NumericalDistribution(central_value=central_value, x=x, y=y)

def _convolve_multivariate_gaussian_numerical(mvgaussian,
                                              mvnumerical,
                                              central_values='same'):
    assert isinstance(mvgaussian, MultivariateNormalDistribution), \
        "mvgaussian must be a single instance of MultivariateNormalDistribution"
    assert isinstance(mvnumerical, MultivariateNumericalDistribution), \
        "mvgaussian must be a single instance of MultivariateNumericalDistribution"
    nsteps = max(200, *[len(x) for x in mvnumerical.xi])
    xi = np.zeros((len(mvnumerical.xi), nsteps))
    for i, x in enumerate(mvnumerical.xi):
        # enlarge the support
        cvn = mvnumerical.central_value[i]
        cvg = mvgaussian.central_value[i]
        supp = [s[i] for s in mvgaussian.support]
        x_max = cvn + (x[-1] - cvn) + (supp[-1] - cvn) +  np.mean(x) - cvg
        x_min = cvn + (x[0] - cvn) + (supp[0] - cvn) +  np.mean(x) - cvg
        xi[i] = np.linspace(x_min, x_max, nsteps)
    xi_grid = np.array(np.meshgrid(*xi, indexing='ij'))
    # this will transpose from shape (0, 1, 2, ...) to (1, 2, ..., 0)
    xi_grid = np.transpose(xi_grid, tuple(range(1, xi_grid.ndim)) + (0,))
    y_num = np.exp(mvnumerical.logpdf(xi_grid))
    # shift Gaussian to the mean of the support
    xi_grid = xi_grid - np.array([np.mean(x) for x in xi]) + np.array(mvgaussian.central_value)
    y_gauss = np.exp(mvgaussian.logpdf(xi_grid))
    f = scipy.signal.fftconvolve(y_num, y_gauss, mode='same')
    f[f < 0] = 0
    f = f/f.sum()
    if central_values == 'sum':
        # shift back
        xi = (xi.T + np.array(mvgaussian.central_value)).T
    return MultivariateNumericalDistribution(xi, f)


# this dictionary is used for parsing low-level distribution definitions
# in YAML files. A string name is associated to every (relevant) distribution.
class_from_string = {
 'delta': DeltaDistribution,
 'uniform': UniformDistribution,
 'normal': NormalDistribution,
 'asymmetric_normal': AsymmetricNormalDistribution,
 'half_normal': HalfNormalDistribution,
 'gaussian_upper_limit': GaussianUpperLimit,
 'gamma': GammaDistribution,
 'gamma_positive': GammaDistributionPositive,
 'gamma_upper_limit': GammaUpperLimit,
 'general_gamma_upper_limit': GeneralGammaUpperLimit,
 'numerical': NumericalDistribution,
 'multivariate_normal': MultivariateNormalDistribution,
 'multivariate_numerical': MultivariateNumericalDistribution,
 'gaussian_kde': GaussianKDE,
}
