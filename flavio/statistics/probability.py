import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.signal
import math
from flavio.math.functions import normal_logpdf, normal_pdf

########## ProbabilityDistribution Class ##########


class ProbabilityDistribution(object):
    """Common base class for all probability distributions"""

    def __init__(self, central_value, support):
        self.central_value = central_value
        self.support = support
        self._x68 = 0.6826894921370859  # 68.2... %

    def get_central(self):
        return self.central_value


class UniformDistribution(ProbabilityDistribution):

    def __init__(self, central_value, half_range):
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

    @property
    def error_left(self):
        """Return the lower error"""
        return self._x68 * self.half_range

    @property
    def error_right(self):
        """Return the upper error"""
        return self._x68 * self.half_range


class DeltaDistribution(ProbabilityDistribution):

    def __init__(self, central_value):
        super().__init__(central_value, support=(central_value, central_value))

    def get_random(self, size=None):
        if size is None:
            return self.central_value
        else:
            return self.central_value * np.ones(size)

    def logpdf(self, x):
        if x == self.central_value:
            return 0.
        else:
            return -np.inf

    @property
    def error_left(self):
        return 0

    @property
    def error_right(self):
        return 0


class NormalDistribution(ProbabilityDistribution):

    def __init__(self, central_value, standard_deviation):
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

    @property
    def error_left(self):
        """Return the lower error"""
        return self.standard_deviation

    @property
    def error_right(self):
        """Return the upper error"""
        return self.standard_deviation


class AsymmetricNormalDistribution(ProbabilityDistribution):

    def __init__(self, central_value, right_deviation, left_deviation):
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

    @property
    def error_left(self):
        """Return the lower error"""
        return self.left_deviation

    @property
    def error_right(self):
        """Return the upper error"""
        return self.right_deviation


class HalfNormalDistribution(ProbabilityDistribution):

    def __init__(self, central_value, standard_deviation):
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

    @property
    def error_left(self):
        """Return the lower error"""
        if self.standard_deviation >= 0:
            return 0
        else:
            return -self.standard_deviation  # return a positive value!

    @property
    def error_right(self):
        """Return the upper error"""
        if self.standard_deviation <= 0:
            return 0
        else:
            return self.standard_deviation


class GaussianUpperLimit(HalfNormalDistribution):

    def __init__(self, limit, confidence_level):
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
        return limit * scipy.stats.norm.ppf(0.5 + confidence_level / 2.)


class NumericalDistribution(ProbabilityDistribution):

    def __init__(self, x, y, central_value=None):
        if central_value is not None:
            if x[0] <= central_value <= x[-1]:
                super().__init__(central_value=central_value,
                                 support=(x[0], x[-1]))
            else:
                raise ValueError("Central value must be within range provided")
        else:
            mode = x[np.argmax(y)]
            super().__init__(central_value=mode, support=(x[0], x[-1]))
        _x_range = (x[-1] - x[0])
        _bin_width = _x_range / len(x)
        _y_norm = y / sum(y) / _bin_width  # normalize PDF to 1
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            self.logpdf_interp = scipy.interpolate.interp1d(x, np.log(_y_norm),
                                                            fill_value=-np.inf, bounds_error=False)
        _cdf = np.cumsum(_y_norm) * _bin_width
        # adapt the borders of the PPF to be 0 and 1
        _cdf[0] = 0.
        _cdf[-1] = 1.
        self.ppf_interp = scipy.interpolate.interp1d(_cdf, x)
        self.cdf_interp = scipy.interpolate.interp1d(x, _cdf)

    def get_random(self, size=None):
        r = np.random.uniform(size=size)
        return self.ppf_interp(r)

    def logpdf(self, x):
        return self.logpdf_interp(x)

    @property
    def error_left(self):
        """Return the lower error defined such that it contains 68% of the
        probability below the central value"""
        cdf_central = self.cdf_interp(self.central_value)
        return self.central_value - self.ppf_interp(cdf_central * (1 - self._x68))

    @property
    def error_right(self):
        """Return the upper error defined such that it contains 68% of the
        probability above the central value"""
        cdf_central = self.cdf_interp(self.central_value)
        return self.ppf_interp(cdf_central + (1 - cdf_central) * self._x68) - self.central_value

    @classmethod
    def from_pd(cls, pd, nsteps=1000):
        _x = np.linspace(pd.support[0], pd.support[-1], nsteps)
        _y = np.exp(pd.logpdf(_x))
        return cls(central_value=pd.central_value, x=_x, y=_y)


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
        super().__init__(central_value, support=None)
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
            # if n of N parameters are to be exluded, x should have length N-n
            if not len(x) + np.asarray(exclude).size == len(self.central_value):
                raise ValueError("Dimensions do not match")
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
                return _dist_ex.logpdf(x[0])
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

    @property
    def error_left(self):
        """Return the lower errors"""
        return self.err

    @property
    def error_right(self):
        """Return the upper errors"""
        return self.err


class MultivariateNumericalDistribution(ProbabilityDistribution):
    """A multivariate distribution with PDF specified numerically."""

    def __init__(self, xi, y, central_value=None):
        """Initialize a multivariate numerical distribution.

        Parameters:

        - `xi`: for an N-dimensional distribution, a list of N 1D arrays
          specifiying the grid in N dimensions. The 1D arrays must contain
          real, evenly spaced values in strictly ascending order (but the
          spacing can be different for different dimensions).
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
        self.xi = xi
        self.y = y
        if central_value is not None:
            super().__init__(central_value=central_value, support=None)
        else:
            # if no central value is specified, set it to the mode
            mode_index = (slice(None),) + np.unravel_index(y.argmax(), y.shape)
            mode = np.asarray(np.meshgrid(*xi, indexing='ij'))[mode_index]
            super().__init__(central_value=mode, support=None)
            _bin_volume = np.prod([x[1] - x[0] for x in xi])
            _y_norm = y / np.sum(y) / _bin_volume  # normalize PDF to 1
        # ignore warning from log(0)=-np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            self.logpdf_interp = scipy.interpolate.RegularGridInterpolator(xi, np.log(_y_norm),
                                                                           fill_value=-np.inf, bounds_error=False)
        # the following is needed for get_random: intialize to None
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

        Note: the exclude parameter is not implemented yet.
        """
        if exclude is not None:
            raise NotImplementedError(
                "Excluding individual parameters from multivariate numerical distributions not implemented")
        return self.logpdf_interp(x)[0]

    @property
    def error_left(self):
        raise NotImplementedError(
            "1D errors not implemented for multivariate numerical distributions")

    @property
    def error_right(self):
        raise NotImplementedError(
            "1D errors not implemented for multivariate numerical distributions")


# Auxiliary functions

def convolve_distributions(probability_distributions):
    """Combine a set of univariate probability distributions.

    This function is meant for combining uncertainties on a single parameter/
    observable. As an argument, it takes a list of probability distributions
    that all have the same central value. It returns their convolution, but
    with location equal to the original central value.

    At present, this function is only implemented for univariate normal
    distributions.
    """
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    central_value = probability_distributions[
        0].central_value  # central value of the first dist
    try:
        float(central_value)
    except:
        raise AssertionError(
            "Combination only implemented for univariate distributions")
    assert all(p.central_value == central_value for p in probability_distributions), \
        "Distrubtions must all have the same central value"
    # all normal dists
    gaussians = [p for p in probability_distributions if isinstance(
        p, NormalDistribution)]
    # let's alrady combined the normal distributions into 1
    if gaussians:
        gaussian = _convolve_gaussians(gaussians)
    # all delta dists -  they can be ignored!
    deltas = [p for p in probability_distributions if isinstance(
        p, DeltaDistribution)]
    # all other univariate dists
    others = list(set(probability_distributions) -
                  set(gaussians) - set(deltas))
    if not others and not gaussians:
        # if there is only a delta (or more than one), just return it
        return deltas[0]
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
        return _convolve_numerical(numerical)


def _convolve_gaussians(probability_distributions):
    assert all(isinstance(p, NormalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NormalDistribution"
    central_value = probability_distributions[
        0].central_value  # central value of the first dist
    assert all(p.central_value == central_value for p in probability_distributions), \
        "Distrubtions must all have the same central value"
    sigmas = np.array(
        [p.standard_deviation for p in probability_distributions])
    sigma = math.sqrt(np.sum(sigmas**2))
    return NormalDistribution(central_value=central_value, standard_deviation=sigma)


def _convolve_numerical(probability_distributions, nsteps=1000):
    assert all(isinstance(p, NumericalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NumericalDistribution"
    central_value = probability_distributions[
        0].central_value  # central value of the first dist
    assert all(p.central_value == central_value for p in probability_distributions), \
        "Distrubtions must all have the same central value"
    # the combined support is the one including all individual supports
    supports = np.array([p.support for p in probability_distributions])
    support = (supports[:, 0].min(), supports[:, 1].max())
    delta = (support[1] - support[0]) / (nsteps - 1)
    x = np.linspace(support[0], support[1], nsteps)
    # position of the central value
    n_x_central = math.floor((central_value - support[0]) / delta)
    y = None
    for pd in probability_distributions:
        y1 = np.exp(pd.logpdf(x)) * delta
        if y is None:
            # first step
            y = y1
        else:
            # convolution
            y = scipy.signal.fftconvolve(y, y1, 'full')
            # cut out the convolved signal at the right place
            y = y[n_x_central:nsteps + n_x_central]
    return NumericalDistribution(central_value=central_value, x=x, y=y)
