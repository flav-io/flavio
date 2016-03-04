import numpy as np
import scipy.stats
import math


########## ProbabilityDistribution Class ##########
class ProbabilityDistribution(object):
   """Common base class for all probability distributions"""

   def __init__(self, central_value):
      self.central_value = central_value

   def get_central(self):
      return self.central_value

    # here we define the __hash__ and __eq__ methods to be able to use
    # instances as dictionary keys.

   def __hash__(self):
      return id(self)

   def __eq__(self, other):
      return id(self) == id(other)



class UniformDistribution(ProbabilityDistribution):

   def __init__(self, central_value, half_range):
      super().__init__(central_value)
      self.half_range = half_range
      self.range = (self.central_value - self.half_range,
                    self.central_value + self.half_range)

   def get_random(self, size=None):
      return np.random.uniform(self.range[0], self.range[1], size)

   def logpdf(self, x):
       if x < self.range[0] or x >= self.range[1]:
           return 0
       else:
           return -math.log(2*self.half_range)

class DeltaDistribution(ProbabilityDistribution):
   def __init__(self, central_value):
      super().__init__(central_value)

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

class NormalDistribution(ProbabilityDistribution):

   def __init__(self, central_value, standard_deviation):
      super().__init__(central_value)
      self.standard_deviation = standard_deviation

   def get_random(self, size=None):
      return np.random.normal(self.central_value, self.standard_deviation, size)

   def logpdf(self, x):
       return scipy.stats.norm.logpdf(x, self.central_value, self.standard_deviation)

class AsymmetricNormalDistribution(ProbabilityDistribution):

   def __init__(self, central_value, right_deviation, left_deviation):
      super().__init__(central_value)
      if right_deviation < 0 or left_deviation < 0:
          raise ValueError("Left and right standard deviations must be positive numbers")
      self.right_deviation = right_deviation
      self.left_deviation = left_deviation

   def get_random(self, size=None):
        r = np.random.uniform()
        a = abs(self.left_deviation/(self.right_deviation+self.left_deviation))
        if  r > a:
            x = abs(np.random.normal(0,self.right_deviation))
            return self.central_value + x
        else:
            x = abs(np.random.normal(0,self.left_deviation))
            return self.central_value - x

   def logpdf(self, x):
       # values of the PDF at the central value
       p_right = scipy.stats.norm.pdf(self.central_value, self.central_value, self.right_deviation)
       p_left = scipy.stats.norm.pdf(self.central_value, self.central_value, self.left_deviation)
       if x < self.central_value:
           # left-hand side: scale factor
           r = 2*p_right/(p_left+p_right)
           return math.log(r) + scipy.stats.norm.logpdf(x, self.central_value, self.left_deviation)
       else:
           # left-hand side: scale factor
           r = 2*p_left/(p_left+p_right)
           return math.log(r) + scipy.stats.norm.logpdf(x, self.central_value, self.right_deviation)

class MultivariateNormalDistribution(ProbabilityDistribution):

   def __init__(self, central_value, covariance):
      super().__init__(central_value)
      assert np.all(np.linalg.eigvals(covariance) > 0), "The covariance matrix is not positive definite!" + str(covariance)
      self.covariance = covariance

   def get_random(self, size=None):
      return np.random.multivariate_normal(self.central_value, self.covariance, size)

   def logpdf(self, x):
       return scipy.stats.multivariate_normal.logpdf(x, self.central_value, self.covariance)




# Auxiliary functions

def combine_distributions(probability_distributions):
    """Combine a set of univariate probability distributions.

    This function is meant for combining uncertainties on a single parameter/
    observable. As an argument, it takes a list of Gaussians that all have
    the same mean. It returns their convolution, but with location equal to the
    original mean.

    This should be generalized to other kinds of distributions in the future...
    """
    # if there's just one: return it immediately
    if len(probability_distributions) == 1:
        return probability_distributions[0]
    central_value = probability_distributions[0].central_value # central value of the first dist
    assert isinstance(central_value, float), "Combination only implemented for univariate distributions"
    gaussians = []
    for pd in probability_distributions:
        if pd.central_value != central_value:
            raise ValueError("All distributions to be combined must have the same central value")
        if isinstance(pd, DeltaDistribution):
            # delta distributions (= no error) can be skipped
            continue
        elif isinstance(pd, NormalDistribution):
            gaussians.append(pd)
        else:
            raise ValueError("Combination only implemented for normal distributions")
    err_squared = sum([pd.standard_deviation**2 for pd in gaussians])
    return NormalDistribution(central_value=central_value, standard_deviation=math.sqrt(err_squared))
