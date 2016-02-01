import numpy as np

########## Parameter Class ##########
class Parameter:
   'Common base class for all parameters'
   def __init__(self, name):
      self.name = name
      self.constraints_list = []

   def set_description(self, description):
      self.description = description

   def get_description(self):
      return self.description

   def add_constraint(self, constraint):
      if not self.constraints_list:
         self.constraints_list.append(constraint)
      else:
         if self.constraints_list[0].central_value == constraint.central_value:
            self.constraints_list.append(constraint)
         else:
            print('The central values of all constraints on one parameter should be equal!')

   def remove_constraints(self):
      self.constraints_list = []

   def get_central(self):
      if not self.constraints_list:
           print('No constraints applied to parameter')
      else:
         return self.constraints_list[0].central_value

   def get_random(self, size=None):
      c = self.get_central()
      r = [u.get_random(size) - c for u in self.constraints_list]
      return np.sum(r, axis=0) + c



########## ProbabilityDistribution Class ##########
class ProbabilityDistribution:
   'Common base class for all ProbabilityDistribution'

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
