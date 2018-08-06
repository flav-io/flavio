"""Functions for working with YAML files."""

import yaml
from collections import OrderedDict


def represent_dict_order(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)
