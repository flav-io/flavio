"""Functions for working with YAML files."""

import yaml
from collections import OrderedDict
import os
import warnings
import errno


def represent_dict_order(self, data):
    return self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)

def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))
yaml.add_constructor('tag:yaml.org,2002:map', dict_constructor)
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:map', dict_constructor)


class SafeIncludeLoader(yaml.SafeLoader):
    """PyYAML loader supporting the `!include` and `!include_merge_list`
    constructors."""
    def __init__(self, stream):
        try:
            self._root = os.path.dirname(stream.name)
        except AttributeError:
            # this happens when `stream` is a string
            self._root = None
        super().__init__(stream)

    def include(self, node):
        file = self.construct_scalar(node)
        filename = self._get_filename(file)
        if filename is None and self._root is None:
            raise ValueError("'{}' cannot be included. If the YAML input is a string, '!include' only supports absolute paths.".format(file))
        with open(filename, 'r') as f:
            return yaml.load(f, self.__class__)

    def include_merge_list(self, node):
        if self._root is None:
            warnings.warn("If the YAML input is a string, '!include_merge_list' only supports absolute paths.")
        files = self.construct_sequence(node)
        a = []
        for file in files:
            filename = self._get_filename(file)
            if filename is None:
                a.append(file)
            else:
                try:
                    with open(filename, 'r') as f:
                        a += yaml.load(f, self.__class__)
                except OSError as e:
                    if e.errno in [errno.ENOENT, errno.EINVAL, errno.EISDIR]:
                        # ENOENT: FileNotFoundError
                        # EINVAL: Invalid argument
                        # EISDIR: IsADirectoryError
                        a.append(file)
                    else:
                        raise
        return a

    def _get_filename(self, file):
        try:
            if os.path.isabs(file):
                return file
            elif self._root is not None:
                return os.path.join(self._root, file)
            else:
                return None
        except TypeError:
            return None




SafeIncludeLoader.add_constructor('!include',
                                  SafeIncludeLoader.include)
SafeIncludeLoader.add_constructor('!include_merge_list',
                                  SafeIncludeLoader.include_merge_list)

def load_include(*args, **kwargs):
    return yaml.load(*args, Loader=SafeIncludeLoader, **kwargs)
