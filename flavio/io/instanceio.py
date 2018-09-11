"""Functions to load and dump class instances from and to YAML dictionaries
or streams."""

import flavio
import voluptuous as vol
import yaml
import base64
import dill
import warnings
from collections import OrderedDict


class YAMLLoadable(object):
    """Base class for objects that can be loaded and dumped from and to
    a dict or YAML stream."""

    # these class attributes should be overwritten by child classes
    _input_schema_dict = {}
    _output_schema_dict = {}

    @classmethod
    def input_schema(cls):
        return vol.Schema(cls._input_schema_dict, extra=vol.ALLOW_EXTRA)

    @classmethod
    def output_schema(cls):
        return vol.Schema(cls._output_schema_dict, extra=vol.REMOVE_EXTRA)

    @classmethod
    def load_dict(cls, d):
        """Instantiate an object from a YAML dictionary."""
        schema = cls.input_schema()
        return cls(**schema(d))

    @classmethod
    def load(cls, f):
        """Instantiate an object from a YAML string or stream."""
        d = flavio.io.yaml.load_include(f)
        return cls.load_dict(d)

    def get_yaml_dict(self):
        """Dump the object to a YAML dictionary."""
        d = self.__dict__.copy()
        schema = self.output_schema()
        d = schema(d)
        # remove NoneTypes and empty lists
        d = {k: v for k, v in d.items() if v is not None and v != []}
        return d

    def dump(self, stream=None, **kwargs):
        """Dump the objectto a YAML string or stream."""
        d = self.get_yaml_dict(**kwargs)
        return yaml.dump(d, stream=stream, **kwargs)


def coerce_observable_tuple(obs):
    """Force an arbitrary observable representation into the tuple representation."""
    return flavio.Observable.argument_format(obs, format='tuple')


def coerce_observable_dict(obs):
    """Force an arbitrary observable representation into the dict representation."""
    return flavio.Observable.argument_format(obs, format='dict')


def coerce_par_obj(par_obj_dict):
    """Coerce a dictionary of parameter constraints into a `ParameterConstraints`
    instance taking `flavio.default_parameters` as starting point"""
    par_obj = flavio.default_parameters.copy()
    return flavio.ParameterConstraints.from_yaml_dict(par_obj_dict,
                                                      instance=par_obj)


def ensurelist(v):
    """Coerce NoneType to empty list, wrap non-list in list."""
    if isinstance(v, list):
        return v
    elif v is None:
        return []
    else:
        raise ValueError("Unexpected form of list: {}".format(v))


def fencode(f):
    if f is None:
        return None
    return {'pickle': base64.b64encode(dill.dumps(f)).decode('utf-8')}


def get_par_diff(par_obj):
    """Return a dictionary representation of a ParameterConstraints instance
    that only contains constraints that are not identical to ones in
    `default_parameters`."""
    dict_default = flavio.default_parameters.get_yaml_dict()
    dict_par = par_obj.get_yaml_dict()
    return [c for c in dict_par if c not in dict_default]


def wc_function_factory(d):
    """Return a Wilson coefficient function suitable for the `fit_wc_function`
    argument starting from a dictionary.

    There are three allowed forms. First form: simply taking the real values
    of WCxf Wilson coefficients:

    ```{'args': ['C9_bsmumu', 'C10_bsmumu']}```

    which is equivalent to

    ```lambda C9_bsmumu, C10_bsmumu: {'C9_bsmumu': C9_bsmumu, 'C10_bsmumu': C10_bsmumu}```

    Second form: giving executable strings for each return key.

    ```{'args': ['ReC9', 'ImC9'],
        'return': {'C9_bsmumu': 'ReC9 + 1j * ImC9'}}```

    which is equivalent to

    ```lambda ReC9, ImC9: {'C9_bsmumu': ReC9 + 1j * ImC9}```

    Third form: explicitly giving the Python code.
    The function name is arbitrary. When using a lambda function,
    it must be assigned to a name.

    ```{'code': "def f(C9, C10):\n  return {'C9_bsmumu': 10 * C9, 'C10_bsmumu': 30 * C10}"```
    """
    if 'code' in d:
        s = d['code']
    elif 'pickle' in d:
        return dill.loads(base64.b64decode(d['pickle'].encode('utf-8')))
    elif 'args' not in d:
        raise ValueError("Function dictionary not understood.")
    elif 'return' not in d:
        s = r"""def _f({}):
    return locals()""".format(', '.join(d['args']))
    else:
        s = r"""def _f({}):
    return {{{}}}""".format(', '.join(d['args']), ', '.join(["'{}': {}".format(k, v) for k, v in d['return'].items()]))
    namespace = OrderedDict()
    exec(s, namespace)  # execute string in empty namespace
    namespace.pop('__builtins__', None)  # remove builtins key if exists
    if not namespace:
        warnings.warn("Function dictionary provided but no function found.")
        return None
    print('NS', namespace)
    f = namespace.popitem()[1]  # assume the last variable is the function
    if not callable(f):
        raise ValueError("Function code not understood")
    return f


def list_deduplicate(lst):
    """Remove duplicate elements from a list but keep the order
    (keep the first occuring element of duplicates). List elements must be
    hashable."""
    return list(OrderedDict.fromkeys(lst))
