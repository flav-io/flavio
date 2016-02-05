import yaml
import pkgutil
from flavio.classes import *

default_parameters = Constraints()

f = pkgutil.get_data('flavio', 'data/parameters.yml')
parameters = yaml.load(f)

for parameter_name, value in parameters.items():
    p = Parameter(parameter_name)
    default_parameters.set_constraint(parameter_name, value)
