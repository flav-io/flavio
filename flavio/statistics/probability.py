"""Probability distributions and auxiliary functions to deal with them."""

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d, RegularGridInterpolator
import scipy.signal
import math
from flavio.math.functions import normal_logpdf, normal_pdf
from flavio.statistics.functions import confidence_level
import warnings
import inspect
from collections import OrderedDict
import yaml
import re


def _camel_to_underscore(s):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def string_to_class(string):
    """Get a ProbabilityDistribution subclass from a string. This can
    either be the class name itself or a string in underscore format
    as returned from `class_to_string`."""
    try:
        return eval(string)
    except NameError:
        pass
    for c in ProbabilityDistribution.get_subclasses():
        if c.class_to_string() == string:
            return c
    raise NameError("Distribution " + string + " not found.")

class ProbabilityDistribution(object):
    """Common base class for all probability distributions"""

    def __init__(self, central_value, support):
        self.central_value = central_value
        self.support = support

    @classmethod
    def get_subclasses(cls):
        """Return all subclasses (including subclasses of subclasses)."""
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

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

    @classmethod
    def class_to_string(cls):
        """Get a string name for a given ProbabilityDistribution subclass.

        This converts camel case to underscore and removes the word
        'distribution'.

        Example: class_to_string(AsymmetricNormalDistribution) returns
        'asymmetric_normal'.
        """
        name = _camel_to_underscore(cls.__name__)
        return name.replace('_distribution', '')

    def get_dict(self, distribution=False, iterate=False, arraytolist=False):
        """Get an ordered dictionary with arguments and values needed to
        the instantiate the distribution.

        Optional arguments (default to False):

        - `distribution`: add a 'distribution' key to the dictionary with the
        value being the string representation of the distribution's name
        (e.g. 'asymmetric_normal').
        - `iterate`: If ProbabilityDistribution instances are among the
        arguments (e.g. for KernelDensityEstimate), return the instance's
        get_dict instead of the instance as value.
        - `arraytolist`: convert numpy arrays to lists
        """
        args = inspect.signature(self.__class__).parameters.keys()
        d = self.__dict__
        od = OrderedDict()
        if distribution:
            od['distribution'] = self.class_to_string()
        od.update(OrderedDict((a, d[a]) for a in args))
        if iterate:
            for k in od:
                if isinstance(od[k], ProbabilityDistribution):
                    od[k] = od[k].get_dict(distribution=True)
        if arraytolist:
            for k in od:
                if isinstance(od[k], np.ndarray):
                    od[k] = od[k].tolist()
                if isinstance(od[k], list):
                    for i, x in enumerate(od[k]):
                        if isinstance(x, np.ndarray):
                            od[k][i] = od[k][i].tolist()
        for k in od:
            if isinstance(od[k], int):
                od[k] = int(od[k])
            elif isinstance(od[k], float):
                od[k] = float(od[k])
            if isinstance(od[k], list):
                for i, x in enumerate(od[k]):
                    if isinstance(x, float):
                        od[k][i] = float(od[k][i])
                    elif isinstance(x, int):
                        od[k][i] = int(od[k][i])
        return od

    def get_yaml(self, *args, **kwargs):
        """Get a YAML string representing the dictionary returned by the
        get_dict method.

        Arguments will be passed to `yaml.dump`."""
        od = self.get_dict(distribution=True, iterate=True, arraytolist=True)
        return yaml.dump(od, *args, **kwargs)

    def delta_logpdf(self, x, **kwargs):
        exclude = kwargs.get('exclude', None)
        if exclude is not None:
            d = len(self.central_value)
            cv = [self.central_value[i] for i in range(d) if i not in exclude]
        else:
            cv = self.central_value
        return self.logpdf(x, **kwargs) - self.logpdf(cv, **kwargs)

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

    def __repr__(self):
        return 'flavio.statistics.probability.UniformDistribution' + \
               '({}, {})'.format(self.central_value, self.half_range)

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

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        return confidence_level(nsigma) * self.half_range

    def get_error_right(self, nsigma=1, **kwargs):
        """Return the upper error"""
        return confidence_level(nsigma) * self.half_range


class DiscreteUniformDistribution(ProbabilityDistribution):
    """Distribution with a finite number of integer values having equal probability"""

    def __init__(self, lowest_value, highest_value, central_value=None):
        """Initialize the distribution.

        Parameters:

        - lowest_value: lowest integer value `a`
        - highest_value: highest integer value `b`
        - central_value (optional): integer within [a,b]. Default is int((a+b)/2)
        """
        if central_value and (
            central_value > highest_value
            or central_value < lowest_value
        ):
            raise ValueError('`central_value` must in the interval [`lowest_value`, `highest_value`].')
        central_value = int((lowest_value+highest_value)/2) if central_value is None else int(central_value)
        self.lowest_value = int(lowest_value)
        self.highest_value = int(highest_value)
        self.range = np.arange(self.lowest_value,self.highest_value+1)
        super().__init__(central_value, support=(self.lowest_value, self.highest_value))

    def __repr__(self):
        return (
            'flavio.statistics.probability.DiscreteUniformDistribution'
            f'({self.support[0]}, {self.support[1]}, {self.central_value})'
        )

    def get_random(self, size=None):
        return np.random.randint(self.support[0], self.support[1]+1, size=size)

    def _logpdf(self, x):
        if x in self.range:
            return -np.log(len(self.range))
        else:
            return -np.inf

    def logpdf(self, x):
        _lpvect = np.vectorize(self._logpdf)
        return _lpvect(x)

    def get_error_left(self, *args, **kwargs):
        return 0

    def get_error_right(self, *args, **kwargs):
        return 0


class DeltaDistribution(ProbabilityDistribution):
    """Delta Distrubution that is non-vanishing only at a single point."""

    def __init__(self, central_value):
        """Initialize the distribution.

        Parameters:

        - central_value: point where the PDF does not vanish.
        """
        super().__init__(central_value, support=(central_value, central_value))

    def __repr__(self):
        return 'flavio.statistics.probability.DeltaDistribution' + \
               '({})'.format(self.central_value)

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

    def __repr__(self):
        return 'flavio.statistics.probability.NormalDistribution' + \
               '({}, {})'.format(self.central_value, self.standard_deviation)

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

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        return nsigma * self.standard_deviation

    def get_error_right(self, nsigma=1, **kwargs):
        """Return the upper error"""
        return nsigma * self.standard_deviation


class LogNormalDistribution(ProbabilityDistribution):
    """Univariate log-normal distribution."""

    def __init__(self, central_value, factor):
        r"""Initialize the distribution.

        Parameters:

        - central_value: median of the distribution (neither mode nor mean!).
          Can be positive or negative, but must be nonzero.
        - factor: must be larger than 1. 68% of the probability will be between
          `central_value * factor` and `central_value / factor`.

        The mean and standard deviation of the underlying normal distribution
        correspond to `log(abs(central_value))` and `log(factor)`, respectively.

        Example:

        `LogNormalDistribution(central_value=3, factor=2)`

        corresponds to the distribution of the exponential of a normally
        distributed variable with mean ln(3) and standard deviation ln(2).
        68% of the probability is within 6=3*2 and 1.5=4/2.
        """
        if central_value == 0:
            raise ValueError("Central value must not be zero")
        if factor <= 1:
            raise ValueError("Factor must be bigger than 1")
        self.factor = factor
        self.log_standard_deviation = np.log(factor)
        self.log_central_value = math.log(abs(central_value))
        if central_value < 0:
            self.central_sign = -1
            slim = math.exp(math.log(abs(central_value))
                            - 6 * self.log_standard_deviation)
            super().__init__(central_value,
                             support=(slim, 0))
        else:
            self.central_sign = +1
            slim = math.exp(math.log(abs(central_value))
                            + 6 * self.log_standard_deviation)
            super().__init__(central_value,
                             support=(0, slim))

    def __repr__(self):
        return 'flavio.statistics.probability.LogNormalDistribution' + \
               '({}, {})'.format(self.central_value, self.factor)

    def get_random(self, size=None):
        s = self.central_sign
        return s * np.random.lognormal(self.log_central_value, self.log_standard_deviation, size)

    def logpdf(self, x):
        s = self.central_sign
        return scipy.stats.lognorm.logpdf(s * x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)

    def pdf(self, x):
        s = self.central_sign
        return scipy.stats.lognorm.pdf(s * x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)

    def cdf(self, x):
        if self.central_sign == -1:
            return 1 - scipy.stats.lognorm.cdf(-x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)
        else:
            return scipy.stats.lognorm.cdf(x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)

    def ppf(self, x):
        if self.central_sign == -1:
            return -scipy.stats.lognorm.ppf(1 - x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)
        else:
            return scipy.stats.lognorm.ppf(x, scale=np.exp(self.log_central_value), s=self.log_standard_deviation)

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        cl = confidence_level(nsigma)
        return self.central_value - self.ppf(0.5 - cl/2.)

    def get_error_right(self, nsigma=1, **kwargs):
        """Return the upper error"""
        cl = confidence_level(nsigma)
        return self.ppf(0.5 + cl/2.) - self.central_value


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

    def __repr__(self):
        return 'flavio.statistics.probability.AsymmetricNormalDistribution' + \
               '({}, {}, {})'.format(self.central_value,
                                 self.right_deviation,
                                 self.left_deviation)

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

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        return nsigma * self.left_deviation

    def get_error_right(self, nsigma=1, **kwargs):
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

    def __repr__(self):
        return 'flavio.statistics.probability.HalfNormalDistribution' + \
               '({}, {})'.format(self.central_value, self.standard_deviation)

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
        if np.sign(self.standard_deviation) == -1:
            return 1 - scipy.stats.halfnorm.cdf(-x,
                                                loc=-self.central_value,
                                                scale=-self.standard_deviation)
        else:
            return scipy.stats.halfnorm.cdf(x,
                                            loc=self.central_value,
                                            scale=self.standard_deviation)

    def ppf(self, x):
        if np.sign(self.standard_deviation) == -1:
            return -scipy.stats.halfnorm.ppf(1 - x,
                                            loc=-self.central_value,
                                            scale=-self.standard_deviation)
        else:
            return scipy.stats.halfnorm.ppf(x,
                                            loc=self.central_value,
                                            scale=self.standard_deviation)


    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        if self.standard_deviation >= 0:
            return 0
        else:
            return nsigma * (-self.standard_deviation)  # return a positive value!

    def get_error_right(self, nsigma=1, **kwargs):
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

    def __repr__(self):
        return 'flavio.statistics.probability.GaussianUpperLimit' + \
               '({}, {})'.format(self.limit, self.confidence_level)

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
        support_min = min(self.scipy_dist.ppf(1e-9), mode)
        support_max = self.scipy_dist.ppf(1-1e-9)
        super().__init__(central_value=mode, # the mode
                         support=(support_min, support_max))
        self.a = a
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        return 'flavio.statistics.probability.GammaDistribution' + \
               '({}, {}, {})'.format(self.a, self.loc, self.scale)

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

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        a = self._find_error_cdf(confidence_level(nsigma))
        return self.central_value - self.ppf(a)

    def get_error_right(self, nsigma=1, **kwargs):
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
        support_min = max(min(self.scipy_dist.ppf(1e-9), mode), 0)
        support_max = self.scipy_dist.ppf(1-1e-9*(1-self.scipy_dist.cdf(0)))
        super().__init__(central_value=mode, # the mode
                         support=(support_min, support_max))
        self.a = a
        self.loc = loc
        self.scale = scale
        # scale factor for PDF to account for x>0
        self._pdf_scale = 1/(1 - self.scipy_dist.cdf(0))

    def __repr__(self):
        return 'flavio.statistics.probability.GammaDistributionPositive' + \
               '({}, {}, {})'.format(self.a, self.loc, self.scale)

    def get_random(self, size=None):
        if size is None:
            return self._get_random(size=size)
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

    def get_error_left(self, nsigma=1, **kwargs):
        """Return the lower error"""
        if self.logpdf(0) > self.logpdf(self.ppf(confidence_level(nsigma))):
            # look at a one-sided 1 sigma range. If the PDF at 0
            # is smaller than the PDF at the boundary of this range, it means
            # that the left-hand error is not meaningful to define.
            return self.central_value
        else:
            a = self._find_error_cdf(confidence_level(nsigma))
            return self.central_value - self.ppf(a)

    def get_error_right(self, nsigma=1, **kwargs):
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

class GammaCountingProcess(GammaDistributionPositive):
    r"""Gamma distribution with x restricted to be positive appropriate for
    a positive quantitity obtained from a low-statistics counting experiment,
    e.g. a rare decay rate, given in terms of event counts and a scale factor.
    The diference to `GammaUpperLimit` is that the scale factor has to be given
    directly and is not expressed in terms of an upper limit.
    """
    def __init__(self, *,
                 scale_factor=1,
                 counts_total=None,
                 counts_background=None,
                 counts_signal=None):
        r"""Initialize the distribution.

        Parameters:

        - `scale_factor`: scale factor by which the number of counts is multiplied
          to get the observable of interest.
        - `counts_total`: observed total number (signal and background) of counts.
        - `counts_background`: expected mean number of expected background counts
        - `counts_signal`: mean observed number of signal events

        Of the three parameters `counts_total`, `counts_background`, and
        `counts_signal`, only two must be specified. The third one will
        be determined from the relation

        `counts_total = counts_signal + counts_background`
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor should be positive")
        self.scale_factor = scale_factor
        if counts_total is not None and counts_total < 0:
            raise ValueError("counts_total should be a positive number, zero, or None")
        if counts_background is not None and counts_background < 0:
            raise ValueError("counts_background should be a positive number, zero, or None")
        if [counts_total, counts_signal, counts_background].count(None) == 0:
            # if all three are specified, check the relation holds!
            if abs((counts_total - counts_background - counts_signal)/(counts_total if counts_total != 0 else 1)) > 1e-15:
                raise ValueError("The relation `counts_total = counts_signal + counts_background` is not satisfied")
        if [counts_total, counts_signal, counts_background].count(None) > 1:
            raise ValueError("Of the three parameters `counts_total`, `counts_background`, and `counts_signal`, two must be specified")
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
        super().__init__(
            a = self.counts_total + 1,
            loc = -self.counts_background * self.scale_factor,
            scale = self.scale_factor
        )

    def __repr__(self):
        return ('flavio.statistics.probability.GammaCountingProcess'
                '(scale_factor={}, counts_total={}, counts_signal={})').format(
                    self.scale_factor,
                    self.counts_total,
                    self.counts_signal
                )

class GammaUpperLimit(GammaCountingProcess):
    r"""Gamma distribution with x restricted to be positive appropriate for
    a positive quantitity obtained from a low-statistics counting experiment,
    e.g. a rare decay rate, given an upper limit on x.
    The diference to `GammaCountingProcess` is that a scale factor is determined
    from the upper limit and not specified directly."""

    def __init__(self, *,
                 limit, confidence_level,
                 counts_total=None,
                 counts_background=None,
                 counts_signal=None):
        r"""Initialize the distribution.

        Parameters:

        - limit: upper limit on x, which is proportional (with a positive
          proportionality factor) to the number of signal events.
        - confidence_level: confidence level of the upper limit, i.e. the value
          of the CDF at the limit. Float between 0 and 1. Frequently used values
          are 0.90 and 0.95.
        - `counts_total`: observed total number (signal and background) of counts.
        - `counts_background`: expected mean number of expected background counts
        - `counts_signal`: mean observed number of signal events
        """
        if confidence_level > 1 or confidence_level < 0:
            raise ValueError("Confidence level should be between 0 und 1")
        if limit <= 0:
            raise ValueError("The upper limit should be a positive number")
        self.limit = limit
        self.confidence_level = confidence_level
        dist_unscaled = GammaCountingProcess(
                 scale_factor=1,
                 counts_total=counts_total,
                 counts_background=counts_background,
                 counts_signal=counts_signal)
        limit_unscaled = dist_unscaled.ppf(self.confidence_level)
        # use the value of the limit to determine the scale factor
        scale_factor = self.limit / limit_unscaled
        super().__init__(
            scale_factor=scale_factor,
            counts_total=counts_total,
            counts_background=counts_background)

    def __repr__(self):
        return ('flavio.statistics.probability.GammaUpperLimit'
                '(limit={}, confidence_level={}, counts_total={}, counts_background={})'
               ).format(self.limit,
                        self.confidence_level,
                        self.counts_total,
                        self.counts_background)

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

    def __repr__(self):
        return 'flavio.statistics.probability.NumericalDistribution' + \
               '({}, {})'.format(self.x, self.y)

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

class GeneralGammaDistributionPositive(NumericalDistribution):
    r"""Distribution appropriate for cases in which a strictly positive quantity
    described by a Gamma distribution has an additional Gaussian uncertainty, which
    is specified by a Gaussian standard deviation. The result is a numerical
    distribution obtained from the convolution of a normal distribution
    (with the Gaussian standard deviation) and a gamma distribution, restricted to
    positive values. Note that the convolution is done before applying the scale factor
    """

    def __init__(self, *, a, loc, scale, gaussian_standard_deviation):
        r"""Initialize the distribution.

        The parameters `a`, `loc`, and `scale` are the same as in `GammaDistributionPositive`.
        The parameter `gaussian_standard_deviation` defines a normal distribution that
        is convoluted with the gamma distribution. Note that the convolution is performed
        before the `scale` factor is applied.

        If `gaussian_standard_deviation=0`, it makes more sense to use
        `GammaDistributionPositive`, which is equivalent but analytical rather than
        numerical.
        """
        if loc > 0:
            raise ValueError("loc must be negative or zero")
        self.a = a
        self.loc = loc
        self.scale = scale
        self.gaussian_standard_deviation = gaussian_standard_deviation
        x, y = self._get_xy()
        if self.gaussian_standard_deviation/np.sqrt(self.counts_total+1) < 1/100:
            self.gaussian_standard_deviation = 0
            warnings.warn("For vanishing or very small Gaussian standard deviation, "
                          "it is safer to use GammaDistributionPositive instead of "
                          "GeneralGammaDistributionPositive to avoid numerical "
                          "instability.")
        super().__init__(x=x, y=y)

    def __repr__(self):
        return ('flavio.statistics.probability.GeneralGammaDistributionPositive'
               '(a={}, loc={}, scale={}, gaussian_standard_deviation={})'.format(
                    self.a,
                    self.loc,
                    self.scale,
                    self.gaussian_standard_deviation
                ))

    def _get_xy(self):
        loc_scaled = self.loc/self.scale
        if self.gaussian_standard_deviation == 0:
            # this is a bit pointless as in this case it makes more
            # sense to use GammaDistributionPositive itself
            gamma_unscaled = GammaDistributionPositive(a = self.a,
                                                       loc = loc_scaled,
                                                       scale = 1)
            num_unscaled = NumericalDistribution.from_pd(gamma_unscaled)
        else:
            # define a gamma distribution (with x>loc, not x>0!) and convolve
            # it with a Gaussian
            gamma_unscaled = GammaDistribution(a = self.a,
                                               loc = loc_scaled,
                                               scale = 1)
            norm_bg = NormalDistribution(0, self.gaussian_standard_deviation)
            num_unscaled = convolve_distributions([gamma_unscaled, norm_bg], central_values='sum')
        # now that we have convolved, add the mirrored values from below x=loc and
        # then cut off anything below x=0
        x = num_unscaled.x
        y = num_unscaled.y_norm
        if loc_scaled in x:
            to_mirror = y[x<=loc_scaled][::-1]
            y_pos = y[len(to_mirror)-1:len(to_mirror)*2-1]
            y[len(to_mirror)-1:len(to_mirror)*2-1] += to_mirror[:len(y_pos)]
        else:
            to_mirror = y[x<loc_scaled][::-1]
            y_pos = y[len(to_mirror):len(to_mirror)*2]
            y[len(to_mirror):len(to_mirror)*2] += to_mirror[:len(y_pos)]
        y = y[x >= 0]
        x = x[x >= 0]
        if x[0] != 0:  #  make sure the PDF at 0 exists
            x = np.insert(x, 0, 0.)  # add 0 as first element
            y = np.insert(y, 0, y[0])  # copy first element
        x = x * self.scale
        return x, y

class GeneralGammaCountingProcess(GeneralGammaDistributionPositive):
    r"""Distribution appropriate for a positive quantitity obtained from a
    low-statistics counting experiment, e.g. a rare decay rate.
    The difference to `GammaCountingProcess` is that this class also allows to
    specify an uncertainty on the number of background events. The result
    is a numerical distribution obtained from the convolution of a normal
    distribution (for the background uncertainty) and a gamma distribution,
    restricted to positive values.
    In contrast to `GeneralGammaUpperLimit`, the scale factor (the relational
    between the observable of interest and the raw number of counts) is not
    determined from a limit and a confidence level, but specified explicitly.
    """

    def __init__(self, *,
                 scale_factor=1,
                 counts_total=None,
                 counts_background=None,
                 counts_signal=None,
                 background_std=0):
        r"""Initialize the distribution.

        Parameters:

        - `scale_factor`: scale factor by which the number of counts is multiplied
          to get the observable of interest.
        - `counts_total`: observed total number (signal and background) of counts.
        - `counts_background`: expected mean number of expected background counts
        - `counts_signal`: mean observed number of signal events
        - `background_std`: standard deviation of the expected number of
          background events

        Of the three parameters `counts_total`, `counts_background`, and
        `counts_signal`, only two must be specified. The third one will
        be determined from the relation

        `counts_total = counts_signal + counts_background`

        Note that if `background_variance=0`, it makes more sense to use
        `GammaCountingProcess`, which is equivalent but analytical rather than
        numerical.
        """
        if scale_factor <= 0:
            raise ValueError("Scale factor should be positive")
        self.scale_factor = scale_factor
        if counts_total is not None and counts_total < 0:
            raise ValueError("counts_total should be a positive number, zero, or None")
        if counts_background is not None and counts_background < 0:
            raise ValueError("counts_background should be a positive number, zero, or None")
        if background_std < 0:
            raise ValueError("background_std should be a positive number")
        if [counts_total, counts_signal, counts_background].count(None) == 0:
            # if all three are specified, check the relation holds!
            if abs((counts_total - counts_background - counts_signal)/(counts_total if counts_total != 0 else 1)) > 1e-15:
                raise ValueError("The relation `counts_total = counts_signal + counts_background` is not satisfied")
        if [counts_total, counts_signal, counts_background].count(None) > 1:
            raise ValueError("Of the three parameters `counts_total`, `counts_background`, and `counts_signal`, two must be specified")
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
        self.background_std = background_std
        if self.background_std/np.sqrt(self.counts_total+1) < 1/100:
            self.background_std = 0
            warnings.warn("For vanishing or very small background standard "
                          "deviation, it is safer to use GammaCountingProcess "
                          "instead of GeneralGammaCountingProcess to avoid "
                          "numerical instability.")
        super().__init__(
            a = self.counts_total + 1,
            loc = -self.counts_background * self.scale_factor,
            scale = self.scale_factor,
            gaussian_standard_deviation = background_std
        )

    def __repr__(self):
        return ('flavio.statistics.probability.GeneralGammaCountingProcess'
               '(scale_factor={}, counts_total={}, counts_signal={}, '
               'background_std={})').format(self.scale_factor,
                                                self.counts_total,
                                                self.counts_signal,
                                                self.background_std)


class GeneralGammaUpperLimit(GeneralGammaCountingProcess):
    r"""Distribution appropriate for
    a positive quantitity obtained from a low-statistics counting experiment,
    e.g. a rare decay rate, given an upper limit on x.
    The difference to `GammaUpperLimit` is that this class also allows to
    specify an uncertainty on the number of background events. The result
    is a numerical distribution obtained from the convolution of a normal
    distribution (for the background uncertainty) and a gamma distribution,
    restricted to positive values.
    The only difference to `GeneralGammaDistributionPositive` is that the scale
    factor is determined from the limit and confidence level.
    """

    def __init__(self, *,
                 limit, confidence_level,
                 counts_total=None,
                 counts_background=None,
                 counts_signal=None,
                 background_std=None,
                 background_variance=None):
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
        - `counts_signal`: mean observed number of signal events
        - `background_std`: standard deviation of the expected number of
          background events
        - `background_variance`: alias of `background_std` for backward
          compatibility

        Of the three parameters `counts_total`, `counts_background`, and
        `counts_signal`, only two must be specified. The third one will
        be determined from the relation

        `counts_total = counts_signal + counts_background`

        Note that if `background_std=0`, it makes more sense to use
        `GammaUpperLimit`, which is equivalent but analytical rather than
        numerical.
        """
        if background_variance is not None:
            warnings.warn(f'The argument `background_variance` is deprecated. Use `background_std` instead.', DeprecationWarning, stacklevel=2)
            if background_std is not None and background_std != background_variance:
                raise ValueError('Setting `background_std` and its alias `background_variance` to different values is inconsistent.')
            else:
                background_std = background_variance
        elif background_std is None:
            background_std = 0
        self.background_variance = background_std
        self.limit = limit
        self.confidence_level = confidence_level
        _d_unscaled = GeneralGammaCountingProcess(
            scale_factor=1,
            counts_total=counts_total,
            counts_background=counts_background,
            counts_signal=counts_signal,
            background_std=background_std)
        limit_unscaled = _d_unscaled.ppf(self.confidence_level)
        # use the value of the limit to determine the scale factor
        scale_factor = self.limit / limit_unscaled
        super().__init__(
            scale_factor=scale_factor,
            counts_total=counts_total,
            counts_background=counts_background,
            counts_signal=counts_signal,
            background_std=background_std)

    def __repr__(self):
        return ('flavio.statistics.probability.GeneralGammaUpperLimit'
               '(limit={}, confidence_level={}, counts_total={}, counts_background={}, '
               'background_std={})').format(self.limit,
                                                self.confidence_level,
                                                self.counts_total,
                                                self.counts_background,
                                                self.background_std)


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
        y, x_edges = np.histogram(data, bins=self.n_bins, density=True)
        x = (x_edges[:-1] + x_edges[1:])/2.
        self.y_raw = y
        self.raw_dist = NumericalDistribution(x, y)
        cdist = convolve_distributions([self.raw_dist, self.kernel], 'sum')
        super().__init__(cdist.x, cdist.y)

    def __repr__(self):
        return 'flavio.statistics.probability.KernelDensityEstimate' + \
               '({}, {}, {})'.format(self.data, repr(self.kernel), self.n_bins)

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

    def __repr__(self):
        return 'flavio.statistics.probability.GaussianKDE' + \
               '({}, {}, {})'.format(self.data, self.bandwidth, self.n_bins)


class MultivariateNormalDistribution(ProbabilityDistribution):
    """A multivariate normal distribution.

    Parameters:

    - central_value: the location vector
    - covariance: the covariance matrix
    - standard_deviation: the square root of the variance vector
    - correlation: the correlation matrix

    If the covariance matrix is not specified, standard_deviation and the
    correlation matrix have to be specified.

    If the covariance or correlation matrix is not symmetric, it is assumed that
    only the values above the diagonal are present and the missing values are
    filled in by reflecting the upper triangle at the diagonal.

    Methods:

    - get_random(size=None): get `size` random numbers (default: a single one)
    - logpdf(x, exclude=None): get the logarithm of the probability density
      function. If an iterable of integers is given for `exclude`, the parameters
      at these positions will be removed from the covariance before evaluating
      the PDF, effectively ignoring certain dimensions.

    Properties:

    - error_left, error_right: both return the vector of standard deviations
    """


    def __init__(self, central_value, covariance=None,
                       standard_deviation=None, correlation=None):
        """Initialize PDF instance.

        Parameters:

        - central_value: vector of means, shape (n)
        - covariance: covariance matrix, shape (n,n)
        - standard_deviation: vector of standard deviations, shape (n)
        - correlation: correlation matrix, shape (n,n)
        """
        if covariance is not None:
            covariance = np.array(covariance)
            if not np.allclose(covariance, covariance.T):
                # if the covariance is not symmetric, it is assumed that only the values above the diagonal are present.
                # then: M -> M + M^T - diag(M)
                covariance = covariance + covariance.T - np.diag(np.diag(covariance))
            self.covariance = covariance
            self.standard_deviation = np.sqrt(np.diag(self.covariance))
            self.correlation = self.covariance/np.outer(self.standard_deviation,
                                                        self.standard_deviation)
            np.fill_diagonal(self.correlation, 1.)
        else:
            if standard_deviation is None:
                raise ValueError("You must specify either covariance or standard_deviation")
            self.standard_deviation = np.array(standard_deviation)
            if correlation is None:
                self.correlation = np.eye(len(self.standard_deviation))
            else:
                if isinstance(correlation, (int, float)):
                    # if it's a number, return delta_ij + (1-delta_ij)*x
                    n_dim = len(central_value)
                    self.correlation = np.eye(n_dim) + (np.ones((n_dim, n_dim))-np.eye(n_dim))*float(correlation)
                else:
                    correlation = np.array(correlation)
                    if not np.allclose(correlation, correlation.T):
                        # if the correlation is not symmetric, it is assumed that only the values above the diagonal are present.
                        # then: M -> M + M^T - diag(M)
                        correlation = correlation + correlation.T - np.diag(np.diag(correlation))
                    self.correlation = correlation
            self.covariance = np.outer(self.standard_deviation,
                                       self.standard_deviation)*self.correlation
        super().__init__(central_value, support=np.array([
                    np.asarray(central_value) - 6*self.standard_deviation,
                    np.asarray(central_value) + 6*self.standard_deviation
                    ]))
        # to avoid ill-conditioned covariance matrices, all data are rescaled
        # by the inverse variances
        self.err = np.sqrt(np.diag(self.covariance))
        self.scaled_covariance = self.covariance / np.outer(self.err, self.err)
        assert np.all(np.linalg.eigvals(self.scaled_covariance) >
                      0), "The covariance matrix is not positive definite!" + str(covariance)

    def __repr__(self):
        return 'flavio.statistics.probability.MultivariateNormalDistribution' + \
               '({}, {})'.format(self.central_value, self.covariance)

    def get_random(self, size=None):
        """Get `size` random numbers (default: a single one)"""
        return np.random.multivariate_normal(self.central_value, self.covariance, size)

    def reduce_dimension(self, exclude=None):
        """Return a different instance where certain dimensions, specified by
        the iterable of integers `exclude`, are removed from the covariance.

        If `exclude` contains all indices but one, an instance of
        `NormalDistribution` will be returned.
        """
        if not exclude:
            return self
        # if parameters are to be excluded, construct a
        # distribution with reduced mean vector and covariance matrix
        _cent_ex = np.delete(self.central_value, exclude)
        _cov_ex = np.delete(
            np.delete(self.covariance, exclude, axis=0), exclude, axis=1)
        if len(_cent_ex) == 1:
            # if only 1 dimension remains, can use a univariate Gaussian
            _dist_ex = NormalDistribution(
                central_value=_cent_ex[0], standard_deviation=np.sqrt(_cov_ex[0, 0]))
        else:
            # if more than 1 dimension remains, use a (smaller)
            # multivariate Gaussian
            _dist_ex = MultivariateNormalDistribution(
                central_value=_cent_ex, covariance=_cov_ex)
        return _dist_ex

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
            _dist_ex = self.reduce_dimension(exclude=exclude)
            return _dist_ex.logpdf(x)
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
        self.xi = [np.asarray(x) for x in xi]
        self.y = np.asarray(y)
        for i, x in enumerate(xi):
            if len(x) == 2:
                self.xi[i] = np.linspace(x[0], x[1], self.y.shape[i])
        if central_value is not None:
            super().__init__(central_value=central_value,
                             support=(np.asarray([x[0] for x in self.xi]), np.asarray([x[-1] for x in self.xi])))
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

    def __repr__(self):
        return 'flavio.statistics.probability.MultivariateNumericalDistribution' + \
               '({}, {}, {})'.format([x.tolist() for x in self.xi], self.y.tolist(), list(self.central_value))

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

    def reduce_dimension(self, exclude=None):
        """Return a different instance where certain dimensions, specified by
        the iterable of integers `exclude`, are removed from the covariance.

        If `exclude` contains all indices but one, an instance of
        `NumericalDistribution` will be returned.
        """
        if not exclude:
            return self
        # if parameters are to be excluded, construct a
        # distribution with reduced mean vector and covariance matrix
        try:
            exclude = tuple(exclude)
        except TypeError:
            exclude = (exclude,)
        xi = [x for i,x in enumerate(self.xi) if i not in exclude]
        y = np.amax(self.y_norm, axis=tuple(exclude))
        cv = np.delete(self.central_value, tuple(exclude))
        if len(xi) == 1:
            # if there is just 1 dimension left, use univariate
            dist = NumericalDistribution(xi[0], y, cv)
        else:
            dist = MultivariateNumericalDistribution(xi, y, cv)
        return dist

    def logpdf(self, x, exclude=None):
        """Get the logarithm of the probability density function.

        Parameters:

        - x: vector; position at which PDF should be evaluated
        - exclude: optional; if an iterable of integers is given, the parameters
          at these positions will be ignored by maximizing the likelihood
          along the remaining directions, i.e., they will be "profiled out".
        """
        if exclude is not None:
            # if parameters are to be excluded, construct a temporary
            # distribution with reduced mean vector and covariance matrix
            # and call its logpdf method
            dist = self.reduce_dimension(exclude=exclude)
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
        if  isinstance(pd, cls):
            # nothing to do
            return pd
        _xi = np.array([np.linspace(pd.support[0][i], pd.support[-1][i], nsteps)
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
    others = [p for p in probability_distributions
              if not isinstance(p, NormalDistribution)
              and not isinstance(p, DeltaDistribution)]

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


def _convolve_numerical(probability_distributions, nsteps=10000, central_values='same'):
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


def combine_distributions(probability_distributions):
    """Combine a set of probability distributions by multiplying the PDFs.

    `probability_distributions` must be a list of instances of descendants of
    `ProbabilityDistribution`.
    """
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
        return _combine_distributions_univariate(probability_distributions)
    else:
        return _combine_distributions_multivariate(probability_distributions)

def _combine_distributions_univariate(probability_distributions):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]

    # all delta dists
    deltas = [p for p in probability_distributions if isinstance(
        p, DeltaDistribution)]

    if len(deltas) > 1:
        # for multiple delta dists, check if central values are the same
        cvs = set([p.central_value for p in deltas])
        if len(cvs) > 1:
            raise ValueError("Combining multiple delta distributions with different central values yields zero PDF")
        else:
            return deltas[0]
    elif len(deltas) == 1:
        # for single delta dist, nothing to combine: delta always wins!
        return deltas[0]

    # all normal dists
    gaussians = [p for p in probability_distributions if isinstance(
        p, NormalDistribution)]

    # all other univariate dists
    others = [p for p in probability_distributions
              if not isinstance(p, NormalDistribution)
              and not isinstance(p, DeltaDistribution)]

    # let's combine the normal distributions into 1
    if gaussians:
        gaussian = _combine_gaussians(gaussians)

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
        return _combine_numerical(numerical)


def weighted_average(central_values, standard_deviations):
    """Return the central value and standard deviation of the weighted average
    if a set of normal distributions specified by a list of central values
    and standard deviations"""
    c = np.average(central_values, weights=1 / np.asarray(standard_deviations)**2)
    u = np.sqrt(1 / np.sum(1 / np.asarray(standard_deviations)**2))
    return c, u


def _combine_gaussians(probability_distributions):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, NormalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NormalDistribution"
    central_values = [p.central_value for p in probability_distributions]
    standard_deviations = [p.standard_deviation for p in probability_distributions]
    c, u = weighted_average(central_values, standard_deviations)
    return NormalDistribution(central_value=c, standard_deviation=u)


def _combine_numerical(probability_distributions, nsteps=1000):
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    assert all(isinstance(p, NumericalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of NumericalDistribution"
    supports = np.array([p.support for p in probability_distributions])
    support = (np.max(supports[:, 0]), np.min(supports[:, 1]))
    if support [1] <= support[0]:
        raise ValueError("Numerical distributions to not have overlapping support")
    x = np.linspace(support[0], support[1], nsteps)
    y = np.exp(np.sum([pd.logpdf(x) for pd in probability_distributions], axis=0))
    return NumericalDistribution(x=x, y=y)


def _combine_distributions_multivariate(probability_distributions):
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]

    # all normal dists
    gaussians = [p for p in probability_distributions if isinstance(
        p, MultivariateNormalDistribution)]

    # all other univariate dists
    others = [p for p in probability_distributions
              if not isinstance(p, MultivariateNormalDistribution)]

    # let's combine the normal distributions into 1
    if gaussians:
        gaussian = _combine_multivariate_gaussians(gaussians)

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
        numerical = [MultivariateNumericalDistribution.from_pd(p) for p in to_be_combined]
        return _combine_multivariate_numerical(numerical)


def _combine_multivariate_gaussians(probability_distributions):
    assert all(isinstance(p, MultivariateNormalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of MultivariateNormalDistribution"
    # covariances: [Sigma_1, Sigma_2, ...]
    # means: [x_1, x_2, ...]
    # weights_ [W_1, W_2, ...] where W_i = (Sigma_i)^(-1)
    # weighted covariance is  (W_1 + W_2 + ...)^(-1) = Sigma
    # weigted mean is  Sigma.(W_1.x_1 + W_2.x_2 + ...) = x
    covariances = [d.covariance for d in probability_distributions]
    means = [d.central_value for d in probability_distributions]
    weights = [np.linalg.inv(c) for c in covariances]
    weighted_covariance = np.linalg.inv(np.sum(weights, axis=0))
    weighted_mean = np.dot(weighted_covariance, np.sum(
        [np.dot(weights[i], means[i]) for i in range(len(means))],
        axis=0))
    return MultivariateNormalDistribution(weighted_mean,
                                          covariance=weighted_covariance)


def _combine_multivariate_numerical(probability_distributions, nsteps=200):
    assert all(isinstance(p, MultivariateNumericalDistribution) for p in probability_distributions), \
        "Distributions should all be instances of MultivariateNumericalDistribution"
    supports = np.array([d.support for d  in probability_distributions])
    xi_min = np.max(supports[:, 0], axis=0)
    xi_max = np.min(supports[:, 1], axis=0)
    assert np.all(xi_min < xi_max), \
        """Support of the multivariate distributions vanishes."""
    ndim = len(probability_distributions[0].central_value)
    _xi = np.array([np.linspace(xi_min[i], xi_max[i], nsteps)
                    for i in range(ndim)])
    _xlist = np.array(np.meshgrid(*_xi, indexing='ij')).reshape(ndim, nsteps**ndim).T
    from functools import reduce
    import operator
    _ylist = reduce(operator.mul, [np.exp(d.logpdf(_xlist)) for d in probability_distributions], 1)
    _y = _ylist.reshape(tuple(nsteps for i in range(ndim)))
    return MultivariateNumericalDistribution(_xi, _y)


def dict2dist(constraint_dict):
    r"""Get a list of probability distributions from a list of dictionaries
    (or a single dictionary) specifying the distributions.

    Arguments:

    - constraint_dict: dictionary or list of several dictionaries of the
      form `{'distribution': 'distribution_name', 'arg1': val1, ...}`, where
      'distribution_name' is a string name associated to each probability
      distribution (see `class_from_string`)
      and `'arg1'`, `val1` are argument/value pairs of the arguments of
      the distribution class's constructor (e.g.`central_value`,
      `standard_deviation` for a normal distribution).
    """
    if isinstance(constraint_dict, dict):
        dict_list = [constraint_dict]
    else:
        dict_list = constraint_dict
    pds = []
    def convertv(v):
        # convert v to float if possible
        try:
            return float(v)
        except:
            return v
    for d in dict_list:
        dist = class_from_string[d['distribution']]
        pds.append(dist(**{k: convertv(v) for k, v in d.items() if k!='distribution'}))
    return pds


# this dictionary is used for parsing low-level distribution definitions
# in YAML files. A string name is associated to every (relevant) distribution.
class_from_string = { c.class_to_string(): c
                      for c in ProbabilityDistribution.get_subclasses() }
