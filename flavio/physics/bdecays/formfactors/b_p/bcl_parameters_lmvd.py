import yaml
import pkgutil
import numpy as np
from flavio.classes import Parameter
from flavio.statistics.probability import MultivariateNormalDistribution

def load_parameters(filename, constraints):
    f = pkgutil.get_data('flavio.physics', filename)
    ff_dict = yaml.safe_load(f)
    dict_name = list(ff_dict.keys())[0]
    observables = ff_dict[dict_name]['observables']
    central_values = np.asarray(ff_dict[dict_name]['medians'])
    covariance = np.asarray(ff_dict[dict_name]['covariance'])
    observables = [o.replace('::', '') for o in observables]
    observables_renamed = []
    for o in observables:
        try:
            index = o.index('f')
        except:
            try:
                index = o.index('b')
            except:
                raise ValueError('No f or b in observable name')
        observables_renamed.append(o[:index]+ ' BCL ' + o[index:])

    for parameter_name in observables_renamed:
        try: # check if parameter object already exists
            p = Parameter[parameter_name]
        except: # otherwise, create a new one
            p = Parameter(parameter_name)
        else: # if parameter exists, remove existing constraints
            constraints.remove_constraint(parameter_name)
    if not np.allclose(covariance, covariance.T):
        # if the covariance is not symmetric, it is assumed that only the values above the diagonal are present.
        # then: M -> M + M^T - diag(M)
        covariance = covariance + covariance.T - np.diag(np.diag(covariance))
    constraints.add_constraint(observables_renamed,
            MultivariateNormalDistribution(central_value=central_values, covariance=covariance) )
