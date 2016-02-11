import numpy as np
import re
from .config import config
from collections import OrderedDict
import copy
import scipy.stats
import math

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class NamedInstanceClass(object):
   """Base class for classes that have named instances that can be accessed
   by their name.

   Parameters
   ----------
    - name: string

   Methods
   -------
    - del_instance(name)
        Delete an instance
    - get_instance(name)
        Delete an instance
    - set_description(description)
        Set the description
    - get_description(description)
        Get the description
   """

   def __init__(self, name):
      if not hasattr(self.__class__, '_instances'):
          self.__class__._instances = OrderedDict()
      self.__class__._instances[name] = self
      self.name = name
      self.description = 'No description available.'

   @classmethod
   def get_instance(cls, name):
      return cls._instances[name]

   @classmethod
   def del_instance(cls, name):
      del cls._instances[name]

   def set_description(self, description):
      self.description = description

   def get_description(self):
      return self.description



########## Parameter Class ##########
class Parameter(NamedInstanceClass):

   def __init__(self, name):
      super().__init__(name)


########## Constraints Class ##########
class Constraints(object):

      def __init__(self):
          self._constraints = OrderedDict()
          self._parameters = OrderedDict()

      @property
      def all_parameters(self):
          return list(self._parameters.keys())

      def add_constraint(self, parameters, constraint):
          for num, parameter in enumerate(parameters):
              # check if there is a constraint and what its central value is
              try: # look at the central value of an existing constraint
                  central_value = self.get_central(parameter)
              except: # if the parameter is not constrained yet, no need to check the central value
                  pass
              else: # if the central value of the new constraint is different from an existing constraint, raise an error
                  if np.ravel([constraint.central_value])[num] != central_value:
                      raise ValueError("The central values of all constraints on one parameter/observable must be equal")
              self._parameters.setdefault(parameter,[]).append((num, constraint))
          self._constraints[constraint] = parameters

      def set_constraint(self, parameter, constraint_string):
          pds = constraints_from_string(constraint_string)
          self.remove_constraints(parameter)
          for pd in pds:
              self.add_constraint([parameter], pd)

      def remove_constraints(self, parameter):
          self._parameters[parameter] = []

      def get_central(self, parameter):
          if parameter not in self._parameters.keys():
              raise ValueError('No constraints applied to parameter/observable ' + self.parameter)
          else:
              num, constraint = self._parameters[parameter][0]
              # return the num-th entry of the central value vector
              return np.ravel([constraint.central_value])[num]

      def get_central_all(self):
          return {parameter: self.get_central(parameter) for parameter in self._parameters.keys()}

      def get_random_all(self):
          random_constraints = [constraint.get_random() for constraint in self._constraints.keys()]
          random_dict = {}
          for parameter, constraints in self._parameters.items():
              central_value =  self.get_central(parameter)
              random_dict[parameter] = central_value
              for num, constraint in constraints:
                  idx = list(self._constraints.keys()).index(constraint)
                  random_dict[parameter] += np.ravel([random_constraints[idx]])[num] - central_value
          return random_dict

      def get_logprobability_all(self, par_dict):
          prob_dict = {}
          for constraint, parameters in self._constraints.items():
              x = [par_dict[p] for p in parameters]
              if len(x) == 1:
                  x = x[0]
              prob_dict[constraint] = constraint.logpdf(x)
          return prob_dict

      def copy(self):
          return copy.copy(self)


########## ParameterConstraints Class ##########
class ParameterConstraints(Constraints):

      def __init__(self):
          super().__init__()



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
       return scipy.stats.multivariate_normal.logpdf(x, self.central_value, self.covariance, allow_singular=True)



########## Observable Class ##########
class Observable(NamedInstanceClass):
   """An Observable is something that can be measured experimentally and
   predicted theoretically."""

   def __init__(self, name, arguments=None):
      super().__init__(name)
      self.arguments = arguments
      self.prediction = None

   def set_prediction(self, prediction):
      self.prediction = prediction

   def prediction_central(self, constraints_obj, wc_obj, *args, **kwargs):
      return self.prediction.get_central(constraints_obj, wc_obj, *args, **kwargs)

   def prediction_par(self, par_dict, wc_obj, *args, **kwargs):
      return self.prediction.get_par(par_dict, wc_obj, *args, **kwargs)


########## AuxiliaryQuantity Class ##########
class AuxiliaryQuantity(NamedInstanceClass):
   """An auxiliary quantity is something that can be computed theoretically but
   not measured directly, e.g. some sub-contribution to an amplitude or a form
   factor."""

   def __init__(self, name, arguments=None):
      super().__init__(name)
      self.arguments = arguments

   def get_implementation(self):
      try:
          implementation_name = config['implementation'][self.name]
      except KeyError:
          raise KeyError("No implementation specified for auxiliary quantity " + self.name)
      return Implementation.get_instance(implementation_name)

   def prediction_central(self, wc_obj, *args, **kwargs):
      implementation = self.get_implementation()
      return implementation.get_central(wc_obj, *args, **kwargs)

   def prediction(self, par_dict, wc_obj, *args, **kwargs):
      implementation = self.get_implementation()
      return implementation.get(par_dict, wc_obj, *args, **kwargs)



########## Prediction Class ##########
class Prediction(object):
   """A prediction is the theoretical prediction for an observable."""

   def __init__(self, observable, function):
      try:
          Observable.get_instance(observable)
      except KeyError:
          raise ValueError("The observable " + observable + " does not exist")
      self.observable = observable
      self.function = function
      self.observable_obj = Observable.get_instance(observable)
      self.observable_obj.set_prediction(self)

   def get_central(self, constraints_obj, wc_obj, *args, **kwargs):
      par_dict = constraints_obj.get_central_all()
      return self.function(wc_obj, par_dict, *args, **kwargs)

   def get_par(self, par_dict, wc_obj, *args, **kwargs):
      return self.function(wc_obj, par_dict, *args, **kwargs)


########## Implementation Class ##########
class Implementation(NamedInstanceClass):
   """An implementation is the theoretical prediction for an auxiliary
   quantity."""

   @classmethod
   def show_all(cls):
      all_dict = {}
      for name in cls._instances:
          inst = cls.get_instance(name)
          quant = inst.quantity
          descr = inst.get_description()
          all_dict[quant] = {name: descr}
      return all_dict

   def __init__(self, name, quantity, function):
      super().__init__(name)
      try:
          AuxiliaryQuantity.get_instance(quantity)
      except KeyError:
          raise ValueError("The quantity " + quantity + " does not exist")
      self.quantity = quantity
      self.function = function
      self.quantity_obj = AuxiliaryQuantity.get_instance(quantity)

   def get_central(self, constraints_obj, wc_obj, *args, **kwargs):
      par_dict = constraints_obj.get_central_all()
      return self.function(wc_obj, par_dict, *args, **kwargs)

   def get(self, par_dict, wc_obj, *args, **kwargs):
      return self.function(wc_obj, par_dict, *args, **kwargs)


########## Measurement Class ##########
class Measurement(Constraints, NamedInstanceClass):
      """A (experimental) measurement associates one (or several) probability
      distributions to one (or several) observables. If it contains several
      observables, these can (but do not have to) be correlated.

      To instantiate the class, call Measurement(name) with a string uniquely
      describing the measurement (e.g. 'CMS Bs->mumu 2012').

      To add a constraint (= central vaue(s) and uncertainty(s)), use

      `add_constraint(observables, constraint)`

      where `constraint` is an instance of a descendant of
      ProbabilityDistribution and `observables` is a list of either
       - a string observable name in the case of observables without arguments
       - or a tuple `(name, x_1, ..., x_n)`, where the `x_i` are float values for the
         arguments, of an observable with `n` arguments.
      """

      def __init__(self, name):
          NamedInstanceClass.__init__(self, name)
          Constraints.__init__(self)
          self.inspire = ''
          self.experiment = ''
          self.url = ''


# Auxiliary functions


def errors_from_string(constraint_string):
    """Convert a string like '1.67(3)(5)' or '1.67+-0.03+-0.05' to a dictionary
    of central values errors."""
    try:
        float(constraint_string)
        # if the string represents just a number, return a DeltaDistribution
        return {'central_value': float(constraint_string)}
    except ValueError:
        # first of all, replace dashes (that can come from copy-and-pasting latex) by minuses
        constraint_string = constraint_string.replace('−','-')
        # try again if the number is a float now
        try:
            float(constraint_string)
            return {'central_value': float(constraint_string)}
        except:
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
    errors = {}
    errors['central_value'] = central_value
    errors['symmetric_errors'] = []
    errors['asymmetric_errors'] = []
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
            errors['symmetric_errors'].append(standard_deviation)
    elif pattern_symmetric_err.match(error_string) or pattern_asymmetric_err.match(error_string):
        for err in re.findall(pattern_symmetric_err, error_string):
            errors['symmetric_errors'].append( float(err)*overall_factor )
        for err in re.findall(pattern_asymmetric_err, error_string):
            right_err = float(err[0])*overall_factor
            left_err = float(err[1])*overall_factor
            errors['asymmetric_errors'].append((right_err, left_err))
    return errors


def constraints_from_string(constraint_string):
    """Convert a string like '1.67(3)(5)' or '1.67+-0.03+-0.05' to a list
    of ProbabilityDistribution instances."""
    errors = errors_from_string(constraint_string)
    if 'symmetric_errors' not in errors and 'asymmetric_errors' not in errors:
        return [DeltaDistribution(errors['central_value'])]
    pd = []
    for err in errors['symmetric_errors']:
        pd.append(NormalDistribution(errors['central_value'], err))
    for err_right, err_left in errors['asymmetric_errors']:
        pd.append(AsymmetricNormalDistribution(errors['central_value'], err_right, err_left))
    return pd
