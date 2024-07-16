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
    constraints.add_constraint(observables_renamed,
            MultivariateNormalDistribution(central_value=central_values, covariance=covariance), is_parameter_constraint=True)
