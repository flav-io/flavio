import yaml
import pkgutil
import numpy as np
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution

def load_parameters(filename, constraints):
    f = pkgutil.get_data('flavio.physics', filename)
    ff_dict = yaml.load(f)
    for parameter_name in ff_dict['parameters']:
        try: # check if parameter object already exists
            p = Parameter[parameter_name]
        except: # otherwise, create a new one
            p = Parameter(parameter_name)
        else: # if parameter exists, remove existing constraints
            constraints.remove_constraint(parameter_name)
    covariance = np.outer(ff_dict['uncertainties'], ff_dict['uncertainties'])*ff_dict['correlation']
    if not np.allclose(covariance, covariance.T):
        # if the covariance is not symmetric, it is assumed that only the values above the diagonal are present.
        # then: M -> M + M^T - diag(M)
        covariance = covariance + covariance.T - np.diag(np.diag(covariance))
    constraints.add_constraint(ff_dict['parameters'],
            MultivariateNormalDistribution(central_value=ff_dict['central_values'], covariance=covariance) )
