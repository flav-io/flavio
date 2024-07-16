import yaml
import pkgutil
import numpy as np
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution

def load_parameters(filename, constraints):
    f = pkgutil.get_data('flavio.physics', filename)
    ff_dict = yaml.safe_load(f)
    covariance = np.outer(ff_dict['uncertainties'], ff_dict['uncertainties'])*ff_dict['correlation']
    constraints.add_constraint(ff_dict['parameters'],
            MultivariateNormalDistribution(central_value=ff_dict['central_values'], covariance=covariance), is_parameter_constraint=True)
