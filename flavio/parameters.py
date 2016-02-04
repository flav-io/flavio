import yaml
import pkgutil
from flavio.classes import *


f = pkgutil.get_data('flavio', 'data/parameters.yml')
parameters = yaml.load(f)

for parameter_name, value in parameters.items():
    p = Parameter(parameter_name)
    p.set_constraint(value)
