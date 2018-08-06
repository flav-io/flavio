"""Functions for working with YAML files."""

import yaml
from collections import OrderedDict
import os


def represent_dict_order(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)


class SafeIncludeLoader(yaml.SafeLoader):
    """PyYAML loader supporting the `!include` and `!include_merge_list`
    constructors."""
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            # this happens when `stream` is a string
            self._root = None
        super().__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)

    def include_merge_list(self, node):
        files = self.construct_sequence(node)
        a = []
        for file in files:
            try:
                filename = os.path.join(self._root, file)
                with open(filename, 'r') as f:
                    a += yaml.load(f, self.__class__)
            except (FileNotFoundError, TypeError):
                a.append(file)
        return a


SafeIncludeLoader.add_constructor('!include',
                                  SafeIncludeLoader.include)
SafeIncludeLoader.add_constructor('!include_merge_list',
                                  SafeIncludeLoader.include_merge_list)

def load_include(*args, **kwargs):
    return yaml.load(*args, Loader=SafeIncludeLoader, **kwargs)
