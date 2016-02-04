import numpy as np
import re

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

########## Parameter Class ##########
class Parameter:

   # all instances of Parameter will be saved to this dict in the form
   # {name: <instance>}
   _instances  = {}

   @classmethod
   def get_instance(cls, name):
      return cls._instances[name]

   @classmethod
   def get_central_all(cls):
      """Return the central values of all parameters."""
      return {k: i.get_central() for k, i in cls._instances.items()}

   @classmethod
   def get_random_all(cls):
      """Return random values for all parameters."""
      return {k: i.get_random() for k, i in cls._instances.items()}

   def __init__(self, name):
      if name in Parameter._instances.keys():
          raise ValueError("The parameter " + name + " already exists")
      Parameter._instances[name] = self
      self.name = name
      self._constraints_list = []

   def set_description(self, description):
      self.description = description

   def get_description(self):
      return self.description

   def add_constraint(self, constraint):
      if not self._constraints_list:
         self._constraints_list.append(constraint)
      else:
         if self._constraints_list[0].central_value == constraint.central_value:
            self._constraints_list.append(constraint)
         else:
            raise ValueError("The central values of all constraints on one parameter should be equal!")

   def remove_constraints(self):
      self._constraints_list = []

   def get_central(self):
      if not self._constraints_list:
           raise ValueError('No constraints applied to parameter ' + self.name)
      else:
         return self._constraints_list[0].central_value

   def get_random(self, size=None):
      c = self.get_central()
      r = [u.get_random(size) - c for u in self._constraints_list]
      return np.sum(r, axis=0) + c

   def set_constraint(self, constraint_string):
      pds = constraints_from_string(constraint_string)
      self.remove_constraints()
      for pd in pds:
          self.add_constraint(pd)



########## ProbabilityDistribution Class ##########
class ProbabilityDistribution:
   """Common base class for all probability distributions"""

   def __init__(self, central_value):
      self.central_value = central_value

   def get_central(self):
      return self.central_value



class DeltaDistribution(ProbabilityDistribution):

   def __init__(self, central_value):
      super().__init__(central_value)

   def get_random(self, size=None):
      if size is None:
          return self.central_value
      else:
          return self.central_value * np.ones(size)

class NormalDistribution(ProbabilityDistribution):

   def __init__(self, central_value, standard_deviation):
      super().__init__(central_value)
      self.standard_deviation = standard_deviation

   def get_random(self, size=None):
      return np.random.normal(self.central_value, self.standard_deviation, size)

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

class MultivariateNormalDistribution(ProbabilityDistribution):

   def __init__(self, central_value, covariance):
      super().__init__(central_value)
      self.covariance = covariance

   def get_random(self, size=None):
      return np.random.multivariate_normal(self.central_value, self.covariance, size)



########## Observable Class ##########
class Observable:

   _instances  = {}

   @classmethod
   def get_instance(cls, name):
      return cls._instances[name]

   def __init__(self, name, arguments=None):
      if name in Observable._instances.keys():
          raise ValueError("The observable " + name + " already exists")
      Observable._instances[name] = self
      self.name = name
      self.arguments = arguments
      self.prediction = None

   def set_description(self, description):
      self.description = description

   def get_description(self):
      return self.description

   def set_prediction(self, prediction):
      self.prediction = prediction

   def prediction_central(self, wc_obj, *args, **kwargs):
      return self.prediction.get_central(wc_obj, *args, **kwargs)


########## Prediction Class ##########
class Prediction:

   def __init__(self, observable, function):
      if observable not in Observable._instances.keys():
          raise ValueError("The observable " + observable + " does not exist")
      self.observable = observable
      self.function = function
      self.observable_obj = Observable._instances[observable]
      self.observable_obj.set_prediction(self)

   def get_central(self, wc_obj, *args, **kwargs):
      par_dict = Parameter.get_central_all()
      return self.function(wc_obj, par_dict, *args, **kwargs)



# Auxiliary functions


def constraints_from_string(constraint_string):
    """Convert a string like '1.67(3)(5)' or '1.67+-0.03+-0.05' to a list
    of ProbabilityDistribution instances."""
    try:
        float(constraint_string)
        # if the string represents just a number, return a DeltaDistribution
        return [DeltaDistribution(float(constraint_string))]
    except ValueError:
        pass
    # for strings of the form '1.67(3)(5) 1e-3'
    pattern_brackets = re.compile(r"^\(?\s*(-?\d+\.?\d*)\s*((?:\(\s*\d+\.?\d*\s*\)\s*)+)\)?\s*\*?\s*(?:(?:e|E|1e|1E|10\^)\(?([+-]?\d+)\)?)?$")
    # for strings of the form '(1.67 +- 0.3 +- 0.5) * 1e-3'
    pattern_plusminus = re.compile(r"^\(?\s*(-?\d+\.?\d*)\s*((?:[+-±\\pm]+\s*\d+\.?\d*\s*)+)\)?\s*\*?\s*(?:(?:e|E|1e|1E|10\^)\(?([+-]?\d+)\)?)?$")
    m = pattern_brackets.match(constraint_string)
    if m is None:
        m = pattern_plusminus.match(constraint_string)
    if m is None:
        raise ValueError("Constraint " + constraint_string + " not understood")
    # extracting the central value and overall power of 10
    if m.group(3) is None:
        overall_factor = 1
    else:
        overall_factor = 10**float(m.group(3))
    central_value = m.group(1)
    # number_decimal gives the number of digits after the decimal point
    if len(central_value.split('.')) == 1:
        number_decimal = 0
    else:
        number_decimal = len(central_value.split('.')[1])
    central_value = float(central_value) * overall_factor
    # now, splitting the errors
    error_string = m.group(2)
    pattern_brackets_err = re.compile(r"\(\s*(\d+\.?\d*)\s*\)\s*")
    pattern_symmetric_err = re.compile(r"(?:±|\\pm|\+\-)(\s*\d+\.?\d*)")
    pattern_asymmetric_err = re.compile(r"\+\s*(\d+\.?\d*)\s*\-\s*(\d+\.?\d*)")
    pd = []
    if pattern_brackets_err.match(error_string):
        for err in re.findall(pattern_brackets_err, error_string):
            if not err.isdigit():
                # if isdigit() is false, it means that it is a number
                # with a decimal point (e.g. '1.5'), so no rescaling is necessary
                standard_deviation = float(err)*overall_factor
            else:
                # if the error is just digits, need to rescale it by the
                # appropriate power of 10
                standard_deviation = float(err)*10**(-number_decimal)*overall_factor
            pd.append(NormalDistribution(central_value, standard_deviation))
    elif pattern_symmetric_err.match(error_string) or pattern_asymmetric_err.match(error_string):
        for err in re.findall(pattern_symmetric_err, error_string):
            pd.append(NormalDistribution(central_value, float(err)*overall_factor))
        for err in re.findall(pattern_asymmetric_err, error_string):
            right_err = float(err[0])*overall_factor
            left_err = float(err[1])*overall_factor
            pd.append(AsymmetricNormalDistribution(central_value, right_err, left_err))
    return pd
