import numpy as np

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
            print('The central values of all constraints on one parameter should be equal!')

   def remove_constraints(self):
      self._constraints_list = []

   def get_central(self):
      if not self._constraints_list:
           raise ValueError('No constraints applied to parameter')
      else:
         return self._constraints_list[0].central_value

   def get_random(self, size=None):
      c = self.get_central()
      r = [u.get_random(size) - c for u in self._constraints_list]
      return np.sum(r, axis=0) + c



########## ProbabilityDistribution Class ##########
class ProbabilityDistribution:
   """Common base class for all probability distributions"""

   def __init__(self, central_value):
      self.central_value = central_value

   def get_central(self):
      return self.central_value



class DeltaDistibution(ProbabilityDistribution):

   def __init__(self, central_value):
      super().__init__(central_value)

   def get_random(self, size=None):
      if size is None:
          return self.central_value
      else:
          return self.central_value * np.ones(size)

class NormalDistibution(ProbabilityDistribution):

   def __init__(self, central_value, standard_deviation):
      super().__init__(central_value)
      self.standard_deviation = standard_deviation

   def get_random(self, size=None):
      return np.random.normal(self.central_value, self.standard_deviation, size)

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
          raise ValueError("The observable " + name + " does not exist")
      self.observable = observable
      self.function = function
      self.observable_obj = Observable._instances[observable]
      self.observable_obj.set_prediction(self)

   def get_central(self, wc_obj, *args, **kwargs):
      par_dict = Parameter.get_central_all()
      return self.function(par_dict, wc_obj, *args, **kwargs)
